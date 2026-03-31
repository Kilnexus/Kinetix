const std = @import("std");

pub const MemoryPool = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) MemoryPool {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    pub fn allocator(self: *MemoryPool) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *MemoryPool) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *MemoryPool) void {
        self.arena.deinit();
    }
};

test "memory pool alloc and reset" {
    var pool = MemoryPool.init(std.testing.allocator);
    defer pool.deinit();

    const allocator = pool.allocator();
    const buf = try allocator.alloc(u8, 64);
    @memset(buf, 0xAA);
    pool.reset();

    const buf2 = try allocator.alloc(u8, 16);
    @memset(buf2, 0x55);
}
