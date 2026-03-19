const std = @import("std");

pub const Stats = struct {
    alloc_count: usize,
    free_count: usize,
    resize_count: usize,
    remap_count: usize,
    failed_alloc_count: usize,
    total_allocated_bytes: usize,
    total_freed_bytes: usize,
    live_bytes: usize,
    peak_live_bytes: usize,
    outstanding_allocations: usize,
};

pub const TrackingAllocator = struct {
    child: std.mem.Allocator,
    allocations: std.AutoHashMapUnmanaged(usize, usize) = .empty,
    alloc_count: usize = 0,
    free_count: usize = 0,
    resize_count: usize = 0,
    remap_count: usize = 0,
    failed_alloc_count: usize = 0,
    total_allocated_bytes: usize = 0,
    total_freed_bytes: usize = 0,
    live_bytes: usize = 0,
    peak_live_bytes: usize = 0,

    pub fn init(child: std.mem.Allocator) TrackingAllocator {
        return .{ .child = child };
    }

    pub fn deinit(self: *TrackingAllocator) void {
        self.allocations.deinit(self.child);
        self.* = undefined;
    }

    pub fn allocator(self: *TrackingAllocator) std.mem.Allocator {
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

    pub fn snapshot(self: *const TrackingAllocator) Stats {
        return .{
            .alloc_count = self.alloc_count,
            .free_count = self.free_count,
            .resize_count = self.resize_count,
            .remap_count = self.remap_count,
            .failed_alloc_count = self.failed_alloc_count,
            .total_allocated_bytes = self.total_allocated_bytes,
            .total_freed_bytes = self.total_freed_bytes,
            .live_bytes = self.live_bytes,
            .peak_live_bytes = self.peak_live_bytes,
            .outstanding_allocations = self.allocations.count(),
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.child.rawAlloc(len, alignment, ret_addr) orelse {
            self.failed_alloc_count += 1;
            return null;
        };
        self.allocations.put(self.child, @intFromPtr(ptr), len) catch {
            self.child.rawFree(ptr[0..len], alignment, ret_addr);
            self.failed_alloc_count += 1;
            return null;
        };
        self.alloc_count += 1;
        self.total_allocated_bytes += len;
        self.live_bytes += len;
        self.peak_live_bytes = @max(self.peak_live_bytes, self.live_bytes);
        return ptr;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const ok = self.child.rawResize(memory, alignment, new_len, ret_addr);
        if (!ok) return false;

        const key = @intFromPtr(memory.ptr);
        const old_len = self.allocations.get(key) orelse memory.len;
        self.allocations.put(self.child, key, new_len) catch @panic("tracking allocator metadata OOM on resize");
        self.resize_count += 1;
        if (new_len >= old_len) {
            const delta = new_len - old_len;
            self.total_allocated_bytes += delta;
            self.live_bytes += delta;
            self.peak_live_bytes = @max(self.peak_live_bytes, self.live_bytes);
        } else {
            const delta = old_len - new_len;
            self.total_freed_bytes += delta;
            self.live_bytes -= delta;
        }
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const new_ptr = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;

        const old_key = @intFromPtr(memory.ptr);
        const old_len = self.allocations.fetchRemove(old_key).?.value;
        self.allocations.put(self.child, @intFromPtr(new_ptr), new_len) catch @panic("tracking allocator metadata OOM on remap");
        self.remap_count += 1;
        if (new_len >= old_len) {
            const delta = new_len - old_len;
            self.total_allocated_bytes += delta;
            self.live_bytes += delta;
            self.peak_live_bytes = @max(self.peak_live_bytes, self.live_bytes);
        } else {
            const delta = old_len - new_len;
            self.total_freed_bytes += delta;
            self.live_bytes -= delta;
        }
        return new_ptr;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const len = if (self.allocations.fetchRemove(@intFromPtr(memory.ptr))) |entry| entry.value else memory.len;
        self.free_count += 1;
        self.total_freed_bytes += len;
        self.live_bytes -= len;
        self.child.rawFree(memory, alignment, ret_addr);
    }
};

test "tracking allocator records allocations and frees" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var tracker = TrackingAllocator.init(gpa.allocator());
    defer tracker.deinit();

    const allocator = tracker.allocator();
    const buffer = try allocator.alloc(u8, 64);
    defer allocator.free(buffer);

    const mid = tracker.snapshot();
    try std.testing.expectEqual(@as(usize, 1), mid.alloc_count);
    try std.testing.expectEqual(@as(usize, 64), mid.live_bytes);
    try std.testing.expectEqual(@as(usize, 64), mid.peak_live_bytes);
}
