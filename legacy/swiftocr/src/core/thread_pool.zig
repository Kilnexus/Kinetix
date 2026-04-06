const std = @import("std");

pub const ThreadPool = struct {
    worker_count: usize,

    pub fn init(worker_count: usize) ThreadPool {
        return .{
            .worker_count = if (worker_count == 0) 1 else worker_count,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        _ = self;
    }
};

test "thread pool init fallback" {
    var pool = ThreadPool.init(0);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 1), pool.worker_count);
}
