const std = @import("std");
const io = std.Options.debug_io;

const manifest_candidates = [_][]const u8{
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
};

const codec_candidates = [_][]const u8{
    "codec_browser_onnx_meta.json",
    "MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
};

pub const BundlePaths = struct {
    allocator: std.mem.Allocator,
    model_dir: []u8,
    manifest_path: []u8,
    tts_meta_path: []u8,
    codec_meta_path: []u8,
    tokenizer_model_path: []u8,

    pub fn deinit(self: *BundlePaths) void {
        self.allocator.free(self.model_dir);
        self.allocator.free(self.manifest_path);
        self.allocator.free(self.tts_meta_path);
        self.allocator.free(self.codec_meta_path);
        self.allocator.free(self.tokenizer_model_path);
        self.* = undefined;
    }
};

pub fn findPaths(allocator: std.mem.Allocator, model_dir: []const u8) !?BundlePaths {
    const manifest_path = try findFirstExistingPath(allocator, model_dir, &manifest_candidates) orelse return null;
    errdefer allocator.free(manifest_path);

    const manifest_dir = std.fs.path.dirname(manifest_path) orelse return null;
    const tts_meta_path = try joinIfExists(allocator, manifest_dir, "tts_browser_onnx_meta.json") orelse return null;
    errdefer allocator.free(tts_meta_path);
    const tokenizer_model_path = try joinIfExists(allocator, manifest_dir, "tokenizer.model") orelse return null;
    errdefer allocator.free(tokenizer_model_path);

    const codec_meta_path = try findCodecMetaPath(allocator, model_dir, manifest_dir) orelse return null;
    errdefer allocator.free(codec_meta_path);

    return .{
        .allocator = allocator,
        .model_dir = try allocator.dupe(u8, model_dir),
        .manifest_path = manifest_path,
        .tts_meta_path = tts_meta_path,
        .codec_meta_path = codec_meta_path,
        .tokenizer_model_path = tokenizer_model_path,
    };
}

fn findCodecMetaPath(allocator: std.mem.Allocator, model_dir: []const u8, manifest_dir: []const u8) !?[]u8 {
    if (try joinIfExists(allocator, manifest_dir, "codec_browser_onnx_meta.json")) |path| return path;

    if (try findFirstExistingPath(allocator, model_dir, &codec_candidates)) |path| return path;

    const manifest_parent = std.fs.path.dirname(manifest_dir) orelse return null;
    return try joinIfExists(allocator, manifest_parent, "MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json");
}

fn findFirstExistingPath(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    candidates: []const []const u8,
) !?[]u8 {
    for (candidates) |candidate| {
        if (try joinIfExists(allocator, base_dir, candidate)) |path| return path;
    }
    return null;
}

fn joinIfExists(allocator: std.mem.Allocator, base_dir: []const u8, relative_path: []const u8) !?[]u8 {
    const full_path = try std.fs.path.join(allocator, &.{ base_dir, relative_path });
    errdefer allocator.free(full_path);
    if (!pathExists(full_path)) {
        allocator.free(full_path);
        return null;
    }
    return full_path;
}

fn pathExists(path: []const u8) bool {
    const file = if (std.fs.path.isAbsolute(path))
        std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        std.Io.Dir.cwd().openFile(io, path, .{});
    if (file) |handle| {
        handle.close(io);
        return true;
    } else |_| {
        return false;
    }
}
