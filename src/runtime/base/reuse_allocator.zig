const std = @import("std");

const AllocationMeta = struct {
    len: usize,
    alignment: std.mem.Alignment,
};

const CacheEntry = struct {
    ptr: [*]u8,
    len: usize,
    alignment: std.mem.Alignment,
};

pub const Stats = struct {
    cache_hits: usize,
    cache_misses: usize,
    cached_buffers: usize,
    cached_bytes: usize,
};

pub const ReuseAllocator = struct {
    child: std.mem.Allocator,
    live: std.AutoHashMapUnmanaged(usize, AllocationMeta) = .empty,
    cache: std.ArrayListUnmanaged(CacheEntry) = .empty,
    cache_hits: usize = 0,
    cache_misses: usize = 0,
    cached_bytes: usize = 0,

    pub fn init(child: std.mem.Allocator) ReuseAllocator {
        return .{ .child = child };
    }

    pub fn deinit(self: *ReuseAllocator) void {
        for (self.cache.items) |entry| {
            self.child.rawFree(entry.ptr[0..entry.len], entry.alignment, @returnAddress());
        }
        self.cache.deinit(self.child);
        self.live.deinit(self.child);
        self.* = undefined;
    }

    pub fn allocator(self: *ReuseAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    pub fn snapshot(self: *const ReuseAllocator) Stats {
        return .{
            .cache_hits = self.cache_hits,
            .cache_misses = self.cache_misses,
            .cached_buffers = self.cache.items.len,
            .cached_bytes = self.cached_bytes,
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *ReuseAllocator = @ptrCast(@alignCast(ctx));
        if (findReusableEntry(self, len, alignment)) |cache_index| {
            const entry = self.cache.swapRemove(cache_index);
            self.cached_bytes -= entry.len;
            self.live.put(self.child, @intFromPtr(entry.ptr), .{
                .len = entry.len,
                .alignment = entry.alignment,
            }) catch @panic("reuse allocator metadata OOM");
            self.cache_hits += 1;
            return entry.ptr;
        }

        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.live.put(self.child, @intFromPtr(ptr), .{
            .len = len,
            .alignment = alignment,
        }) catch {
            self.child.rawFree(ptr[0..len], alignment, ret_addr);
            return null;
        };
        self.cache_misses += 1;
        return ptr;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = memory;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = memory;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *ReuseAllocator = @ptrCast(@alignCast(ctx));
        const key = @intFromPtr(memory.ptr);
        const meta = if (self.live.fetchRemove(key)) |entry| entry.value else AllocationMeta{
            .len = memory.len,
            .alignment = alignment,
        };

        if (shouldCache(meta.len, meta.alignment)) {
            if (findReusableEntry(self, meta.len, meta.alignment) != null) {
                self.child.rawFree(memory, alignment, ret_addr);
                return;
            }
            self.cache.append(self.child, .{
                .ptr = memory.ptr,
                .len = meta.len,
                .alignment = meta.alignment,
            }) catch {
                self.child.rawFree(memory, alignment, ret_addr);
                return;
            };
            self.cached_bytes += meta.len;
            return;
        }

        self.child.rawFree(memory, alignment, ret_addr);
    }

    fn findReusableEntry(self: *ReuseAllocator, len: usize, alignment: std.mem.Alignment) ?usize {
        for (self.cache.items, 0..) |entry, index| {
            if (entry.len == len and entry.alignment == alignment) return index;
        }
        return null;
    }

    fn shouldCache(len: usize, alignment: std.mem.Alignment) bool {
        _ = alignment;
        return len >= 4 * 1024 and len <= 8 * 1024 * 1024;
    }
};

test "reuse allocator reuses exact-size buffers" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var reuse = ReuseAllocator.init(gpa.allocator());
    defer reuse.deinit();

    const allocator = reuse.allocator();
    const first = try allocator.alloc(f32, 2048);
    const first_ptr = first.ptr;
    allocator.free(first);

    const second = try allocator.alloc(f32, 2048);
    defer allocator.free(second);

    try std.testing.expectEqual(@intFromPtr(first_ptr), @intFromPtr(second.ptr));
    const stats = reuse.snapshot();
    try std.testing.expectEqual(@as(usize, 1), stats.cache_hits);
}
