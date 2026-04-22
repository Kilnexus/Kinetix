const std = @import("std");
const cli = @import("command.zig");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const parsed = try cli.parse(gpa, args);
    defer parsed.deinit(gpa);

    var stdout_buffer: [4096]u8 = undefined;
    var stderr_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    var stderr_writer = std.Io.File.stderr().writer(init.io, &stderr_buffer);
    const stdout = &stdout_writer.interface;
    const stderr = &stderr_writer.interface;

    cli.run(stdout, stderr, parsed) catch |err| {
        try stderr.print("kinetix error: {t}\n", .{err});
        try stderr.flush();
        return err;
    };
    try stdout.flush();
}
