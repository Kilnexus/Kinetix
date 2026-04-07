const builtin = @import("builtin");
const std = @import("std");

var init_mutex: std.Thread.Mutex = .{};
var init_done: bool = false;
var init_failed: bool = false;
var pool_storage: std.Thread.Pool = undefined;

pub fn get() ?*std.Thread.Pool {
    if (builtin.single_threaded) return null;

    init_mutex.lock();
    defer init_mutex.unlock();

    if (!init_done and !init_failed) {
        pool_storage.init(.{
            .allocator = std.heap.page_allocator,
            .track_ids = false,
        }) catch {
            init_failed = true;
            return null;
        };
        init_done = true;
    }

    return if (init_done) &pool_storage else null;
}
