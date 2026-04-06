const std = @import("std");

pub const ArenaPool = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(backing_allocator: std.mem.Allocator) ArenaPool {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    pub fn allocator(self: *ArenaPool) std.mem.Allocator {
        return self.arena.allocator();
    }

    pub fn reset(self: *ArenaPool) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn deinit(self: *ArenaPool) void {
        self.arena.deinit();
    }
};

test "arena pool alloc and reset" {
    var pool = ArenaPool.init(std.testing.allocator);
    defer pool.deinit();

    const allocator = pool.allocator();
    const buf = try allocator.alloc(u8, 32);
    @memset(buf, 0xAA);
    pool.reset();

    const buf2 = try allocator.alloc(u8, 8);
    @memset(buf2, 0x55);
}
