const std = @import("std");
const common = @import("common.zig");

pub fn runByOutputChannel(
    comptime Context: type,
    ctx: *const Context,
    out_channels: usize,
    thread_count: usize,
    comptime Task: type,
    comptime makeTask: fn (*const Context, usize, usize) Task,
    comptime worker: fn (Task) void,
    comptime runRange: fn (*const Context, usize, usize) common.OpError!void,
) common.OpError!void {
    var threads: [common.max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try runRange(ctx, oc_start, oc_end);
        } else {
            const task = makeTask(ctx, oc_start, oc_end);
            threads[spawned] = std.Thread.spawn(.{}, worker, .{task}) catch {
                try runRange(ctx, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}
