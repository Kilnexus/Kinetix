const std = @import("std");

pub fn run(allocator: std.mem.Allocator) !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const show_help = args.len <= 1 or
        std.mem.eql(u8, args[1], "--help") or
        std.mem.eql(u8, args[1], "-h");

    try stdout.writeAll(
        \\Legacy Zinfer CLI has been retired in this monorepo.
        \\Use the unified Kinetix entrypoint instead.
        \\Example:
        \\  zig build run -- run --model-dir .\models\text\Qwen3-0.6B --operation generate --input "Hello"
        \\
    );

    if (show_help) return;
    return error.InvalidCommand;
}
