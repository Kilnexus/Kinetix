const std = @import("std");

const io = std.Options.debug_io;

pub const Summary = struct {
    builtin_voice_count: usize = 0,
    has_generation_defaults: bool = false,
    has_model_files: bool = false,
};

pub fn loadSummary(allocator: std.mem.Allocator, path: []const u8) !Summary {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidMossTtsManifest;
    const object = parsed.value.object;

    return .{
        .builtin_voice_count = countArrayField(object, "builtin_voices"),
        .has_generation_defaults = object.get("generation_defaults") != null,
        .has_model_files = object.get("model_files") != null,
    };
}

fn countArrayField(object: std.json.ObjectMap, field_name: []const u8) usize {
    const value = object.get(field_name) orelse return 0;
    return switch (value) {
        .array => |items| items.items.len,
        else => 0,
    };
}

test "moss tts manifest summary counts builtin voices" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "browser_poc_manifest.json",
        \\{
        \\  "builtin_voices": ["a", "b"],
        \\  "generation_defaults": {},
        \\  "model_files": {}
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "browser_poc_manifest.json");
    defer std.testing.allocator.free(path);

    const summary = try loadSummary(std.testing.allocator, path);
    try std.testing.expectEqual(@as(usize, 2), summary.builtin_voice_count);
    try std.testing.expect(summary.has_generation_defaults);
    try std.testing.expect(summary.has_model_files);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
