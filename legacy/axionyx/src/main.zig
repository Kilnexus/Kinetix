const std = @import("std");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();

    const gpa = gpa_state.allocator();
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const show_help = args.len <= 1 or
        std.mem.eql(u8, args[1], "--help") or
        std.mem.eql(u8, args[1], "-h");

    try stdout.writeAll(
        \\Legacy Axionyx CLI has been retired in this monorepo.
        \\Use the unified Kinetix entrypoint instead.
        \\Example:
        \\  zig build run -- run --model-dir .\models\vision\compat_yolo11n --operation detect --input .\datasets\vision\archive\images\000_0001.png
        \\
    );

    if (show_help) return;
    return error.InvalidCommand;
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("graph");
    _ = @import("tensor");
    _ = @import("ops");
    _ = @import("weights");
    _ = @import("runtime");
    _ = @import("vision");
}
