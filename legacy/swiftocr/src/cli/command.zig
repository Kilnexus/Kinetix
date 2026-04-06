const std = @import("std");

pub const CommandKind = enum {
    infer,
};

pub const InferArgs = struct {
    model_path: []const u8,
    image_path: []const u8,
};

pub const ParsedCommand = struct {
    kind: CommandKind,
    infer: ?InferArgs,
    help: bool,
    argv_copy: [][]u8,

    pub fn deinit(self: ParsedCommand, allocator: std.mem.Allocator) void {
        for (self.argv_copy) |arg| {
            allocator.free(arg);
        }
        allocator.free(self.argv_copy);
    }
};

pub fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\SwiftOCR CLI
        \\
        \\Usage:
        \\  swiftocr infer --model <path> --image <path>
        \\  swiftocr --help
        \\
    );
}

pub fn parse(allocator: std.mem.Allocator) !ParsedCommand {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var copied = try allocator.alloc([]u8, args.len);
    errdefer {
        for (copied) |arg| allocator.free(arg);
        allocator.free(copied);
    }

    for (args, 0..) |arg, i| {
        copied[i] = try allocator.dupe(u8, arg);
    }

    if (args.len <= 1) {
        return .{
            .kind = .infer,
            .infer = null,
            .help = true,
            .argv_copy = copied,
        };
    }

    if (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h")) {
        return .{
            .kind = .infer,
            .infer = null,
            .help = true,
            .argv_copy = copied,
        };
    }

    if (!std.mem.eql(u8, args[1], "infer")) {
        return error.UnknownCommand;
    }

    var model_path: ?[]const u8 = null;
    var image_path: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingModelPath;
            model_path = copied[i];
            continue;
        }
        if (std.mem.eql(u8, args[i], "--image")) {
            i += 1;
            if (i >= args.len) return error.MissingImagePath;
            image_path = copied[i];
            continue;
        }
        return error.UnknownOption;
    }

    if (model_path == null) return error.MissingModelPath;
    if (image_path == null) return error.MissingImagePath;

    return .{
        .kind = .infer,
        .infer = .{
            .model_path = model_path.?,
            .image_path = image_path.?,
        },
        .help = false,
        .argv_copy = copied,
    };
}

test "parse infer command" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const fake = [_][]const u8{ "swiftocr", "infer", "--model", "a.swm", "--image", "a.ppm" };

    var copied = try allocator.alloc([]u8, fake.len);
    for (fake, 0..) |arg, idx| {
        copied[idx] = try allocator.dupe(u8, arg);
    }

    const parsed = ParsedCommand{
        .kind = .infer,
        .infer = .{
            .model_path = copied[3],
            .image_path = copied[5],
        },
        .help = false,
        .argv_copy = copied,
    };

    try std.testing.expectEqual(CommandKind.infer, parsed.kind);
    try std.testing.expect(parsed.infer != null);
    try std.testing.expectEqualStrings("a.swm", parsed.infer.?.model_path);
    try std.testing.expectEqualStrings("a.ppm", parsed.infer.?.image_path);
}
