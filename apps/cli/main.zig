const std = @import("std");
const cli = @import("command.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();

    const gpa = gpa_state.allocator();
    const parsed = try cli.parse(gpa);
    defer parsed.deinit(gpa);

    var stdout_buffer: [4096]u8 = undefined;
    var stderr_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stdout = &stdout_writer.interface;
    const stderr = &stderr_writer.interface;

    cli.run(stdout, stderr, parsed) catch |err| {
        try stderr.print("kinetix error: {t}\n", .{err});
        try stderr.flush();
        return err;
    };
    try stdout.flush();
}
