const std = @import("std");

pub const paths = @import("paths.zig");
pub const manifest = @import("manifest.zig");
pub const meta = @import("meta.zig");

pub const BundlePaths = paths.BundlePaths;
pub const ManifestSummary = manifest.Summary;
pub const TtsSummary = meta.TtsSummary;
pub const CodecConfig = meta.CodecConfig;

pub const LoadedBundle = struct {
    paths: BundlePaths,
    manifest: ManifestSummary,
    tts: TtsSummary,
    codec: CodecConfig,

    pub fn deinit(self: *LoadedBundle) void {
        self.paths.deinit();
        self.* = undefined;
    }
};

pub fn load(allocator: std.mem.Allocator, model_dir: []const u8) !?LoadedBundle {
    var resolved = try paths.findPaths(allocator, model_dir) orelse return null;
    errdefer resolved.deinit();

    return .{
        .paths = resolved,
        .manifest = try manifest.loadSummary(allocator, resolved.manifest_path),
        .tts = try meta.loadTtsSummary(allocator, resolved.tts_meta_path),
        .codec = try meta.loadCodecConfig(allocator, resolved.codec_meta_path),
    };
}

test "moss tts bundle loader resolves official onnx layout" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makeDir("MOSS-TTS-Nano-100M-ONNX");
    try tmp.dir.makeDir("MOSS-Audio-Tokenizer-Nano-ONNX");
    var tts_dir = try tmp.dir.openDir("MOSS-TTS-Nano-100M-ONNX", .{});
    defer tts_dir.close();
    var codec_dir = try tmp.dir.openDir("MOSS-Audio-Tokenizer-Nano-ONNX", .{});
    defer codec_dir.close();

    try writeTmpFile(tts_dir, "browser_poc_manifest.json",
        \\{
        \\  "builtin_voices": ["speaker_a", "speaker_b"],
        \\  "generation_defaults": {},
        \\  "model_files": {}
        \\}
    );
    try writeTmpFile(tts_dir, "tts_browser_onnx_meta.json",
        \\{
        \\  "model_info": {},
        \\  "session_options": {}
        \\}
    );
    try writeTmpFile(tts_dir, "tokenizer.model", "spm");
    try writeTmpFile(codec_dir, "codec_browser_onnx_meta.json",
        \\{
        \\  "codec_config": {
        \\    "sample_rate": 48000,
        \\    "channels": 2,
        \\    "num_quantizers": 32
        \\  }
        \\}
    );

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var loaded = (try load(std.testing.allocator, root_path)) orelse return error.ExpectedMossTtsBundle;
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.manifest.builtin_voice_count);
    try std.testing.expect(loaded.manifest.has_model_files);
    try std.testing.expect(loaded.tts.has_model_info);
    try std.testing.expect(loaded.tts.has_session_options);
    try std.testing.expectEqual(@as(usize, 48000), loaded.codec.sample_rate);
    try std.testing.expectEqual(@as(usize, 2), loaded.codec.channels);
    try std.testing.expectEqual(@as(usize, 32), loaded.codec.num_quantizers);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
