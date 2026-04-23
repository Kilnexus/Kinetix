const std = @import("std");

const io = std.Options.debug_io;

pub const TtsSummary = struct {
    has_model_info: bool = false,
    has_session_options: bool = false,
};

pub const CodecConfig = struct {
    sample_rate: usize = 0,
    channels: usize = 0,
    num_quantizers: usize = 0,
};

const CodecMeta = struct {
    codec_config: CodecConfig = .{},
};

pub fn loadTtsSummary(allocator: std.mem.Allocator, path: []const u8) !TtsSummary {
    const value = try loadJsonValue(allocator, path);
    defer value.deinit();

    if (value.value != .object) return error.InvalidMossTtsMeta;
    const object = value.value.object;
    return .{
        .has_model_info = object.get("model_info") != null,
        .has_session_options = object.get("session_options") != null,
    };
}

pub fn loadCodecConfig(allocator: std.mem.Allocator, path: []const u8) !CodecConfig {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSliceLeaky(CodecMeta, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
    return parsed.codec_config;
}

fn loadJsonValue(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);
    return try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
}

test "moss tts codec meta parser extracts audio contract" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "codec_browser_onnx_meta.json",
        \\{
        \\  "codec_config": {
        \\    "sample_rate": 48000,
        \\    "channels": 2,
        \\    "num_quantizers": 32
        \\  }
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "codec_browser_onnx_meta.json");
    defer std.testing.allocator.free(path);

    const config = try loadCodecConfig(std.testing.allocator, path);
    try std.testing.expectEqual(@as(usize, 48000), config.sample_rate);
    try std.testing.expectEqual(@as(usize, 2), config.channels);
    try std.testing.expectEqual(@as(usize, 32), config.num_quantizers);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
