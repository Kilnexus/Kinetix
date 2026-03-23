const std = @import("std");

pub const BenchArgs = struct {
    image_path: []const u8,
    iterations: usize,
    warmup: usize,
};

pub const FastBenchArgs = struct {
    image_path: []const u8,
    iterations: usize,
    warmup: usize,
    image_size: usize,
    score_threshold: f32,
};

pub const FastArgs = struct {
    image_path: []const u8,
    image_size: usize,
    score_threshold: f32,
};

pub const ProfileArgs = struct {
    image_path: []const u8,
    image_size: usize,
};

pub const ZeroArgs = struct {
    size: usize,
    json_out_path: ?[]const u8,
    trace_json_out_path: ?[]const u8,
};

pub const ImageArgs = struct {
    image_path: []const u8,
    image_size: usize,
    json_out_path: ?[]const u8,
    trace_json_out_path: ?[]const u8,
};

pub const Command = union(enum) {
    roadmap,
    bench: BenchArgs,
    fastbench: FastBenchArgs,
    fast: FastArgs,
    profile: ProfileArgs,
    zero: ZeroArgs,
    image: ImageArgs,
};

pub const ParsedArgs = struct {
    graph_path: []const u8,
    weights_path: []const u8,
    command: Command,
};

pub fn parseArgs(argv: []const []const u8) ParsedArgs {
    const graph_path = if (argv.len > 1) argv[1] else "artifacts/graph.json";
    const weights_path = if (argv.len > 2) argv[2] else "artifacts/weights.bin";
    const mode_arg = if (argv.len > 3) argv[3] else null;

    if (mode_arg == null) {
        return .{ .graph_path = graph_path, .weights_path = weights_path, .command = .roadmap };
    }

    const value = mode_arg.?;
    if (std.mem.eql(u8, value, "bench")) {
        return .{
            .graph_path = graph_path,
            .weights_path = weights_path,
            .command = .{ .bench = .{
                .image_path = argOr(argv, 4, "data/archive/images/000_0001.png"),
                .iterations = parseIntOr(argAt(argv, 5), 5),
                .warmup = parseIntOr(argAt(argv, 6), 1),
            } },
        };
    }
    if (std.mem.eql(u8, value, "fastbench")) {
        return .{
            .graph_path = graph_path,
            .weights_path = weights_path,
            .command = .{ .fastbench = .{
                .image_path = argOr(argv, 4, "data/archive/images/000_0001.png"),
                .iterations = parseIntOr(argAt(argv, 5), 10),
                .warmup = parseIntOr(argAt(argv, 6), 3),
                .image_size = parseIntOr(argAt(argv, 7), 96),
                .score_threshold = parseFloatOr(argAt(argv, 8), 0.25),
            } },
        };
    }
    if (std.mem.eql(u8, value, "fast")) {
        return .{
            .graph_path = graph_path,
            .weights_path = weights_path,
            .command = .{ .fast = .{
                .image_path = argOr(argv, 4, "data/archive/images/000_0001.png"),
                .image_size = parseIntOr(argAt(argv, 5), 160),
                .score_threshold = parseFloatOr(argAt(argv, 6), 0.25),
            } },
        };
    }
    if (std.mem.eql(u8, value, "profile")) {
        return .{
            .graph_path = graph_path,
            .weights_path = weights_path,
            .command = .{ .profile = .{
                .image_path = argOr(argv, 4, "data/archive/images/000_0001.png"),
                .image_size = parseIntOr(argAt(argv, 5), 160),
            } },
        };
    }
    if (parseIntMaybe(value)) |size| {
        return .{
            .graph_path = graph_path,
            .weights_path = weights_path,
            .command = .{ .zero = .{
                .size = size,
                .json_out_path = argAt(argv, 4),
                .trace_json_out_path = argAt(argv, 5),
            } },
        };
    }

    var image_size: usize = 640;
    var json_out_path: ?[]const u8 = null;
    var trace_json_out_path: ?[]const u8 = null;
    if (argAt(argv, 4)) |arg4| {
        if (parseIntMaybe(arg4)) |parsed| {
            image_size = parsed;
            json_out_path = argAt(argv, 5);
            trace_json_out_path = argAt(argv, 6);
        } else {
            json_out_path = arg4;
            trace_json_out_path = argAt(argv, 5);
        }
    }

    return .{
        .graph_path = graph_path,
        .weights_path = weights_path,
        .command = .{ .image = .{
            .image_path = value,
            .image_size = image_size,
            .json_out_path = json_out_path,
            .trace_json_out_path = trace_json_out_path,
        } },
    };
}

fn argAt(argv: []const []const u8, index: usize) ?[]const u8 {
    return if (index < argv.len) argv[index] else null;
}

fn argOr(argv: []const []const u8, index: usize, default: []const u8) []const u8 {
    return argAt(argv, index) orelse default;
}

fn parseIntMaybe(value: []const u8) ?usize {
    return std.fmt.parseInt(usize, value, 10) catch null;
}

fn parseIntOr(value: ?[]const u8, default: usize) usize {
    return if (value) |text|
        std.fmt.parseInt(usize, text, 10) catch default
    else
        default;
}

fn parseFloatOr(value: ?[]const u8, default: f32) f32 {
    return if (value) |text|
        std.fmt.parseFloat(f32, text) catch default
    else
        default;
}
