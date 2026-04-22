const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const bundle = @import("bundle/locator.zig");
const types = @import("../../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .moss_tts_nano_tts,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const Manifest = struct {
    builtin_voices: ?[]const std.json.Value = null,
    generation_defaults: ?std.json.Value = null,
};

const CodecMeta = struct {
    codec_config: struct {
        sample_rate: usize = 0,
        channels: usize = 0,
        num_quantizers: usize = 0,
    } = .{},
};

const State = struct {
    base: backend_mod.OpenState,
    manifest_path: []u8,
    tts_meta_path: []u8,
    codec_meta_path: []u8,
    tokenizer_model_path: []u8,
    builtin_voice_count: usize = 0,
    sample_rate: usize = 0,
    channels: usize = 0,
    num_quantizers: usize = 0,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.manifest_path);
        allocator.free(self.tts_meta_path);
        allocator.free(self.codec_meta_path);
        allocator.free(self.tokenizer_model_path);
        allocator.free(self.base.model_dir);
        allocator.destroy(self);
    }
};

fn open(
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) !?*anyopaque {
    var resolved = try bundle.findBundlePaths(allocator, model.artifacts.model_dir) orelse return error.ModelManifestNotFound;
    defer resolved.deinit();

    const manifest = try readJsonFile(allocator, Manifest, resolved.manifest_path);
    const codec_meta = try readJsonFile(allocator, CodecMeta, resolved.codec_meta_path);

    const state = try allocator.create(State);
    errdefer allocator.destroy(state);
    state.* = .{
        .base = .{
            .provider_key = .moss_tts_nano_tts,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .manifest_path = try allocator.dupe(u8, resolved.manifest_path),
        .tts_meta_path = try allocator.dupe(u8, resolved.tts_meta_path),
        .codec_meta_path = try allocator.dupe(u8, resolved.codec_meta_path),
        .tokenizer_model_path = try allocator.dupe(u8, resolved.tokenizer_model_path),
        .builtin_voice_count = if (manifest.builtin_voices) |voices| voices.len else 0,
        .sample_rate = codec_meta.codec_config.sample_rate,
        .channels = codec_meta.codec_config.channels,
        .num_quantizers = codec_meta.codec_config.num_quantizers,
    };
    errdefer allocator.free(state.base.model_dir);
    errdefer allocator.free(state.manifest_path);
    errdefer allocator.free(state.tts_meta_path);
    errdefer allocator.free(state.codec_meta_path);
    errdefer allocator.free(state.tokenizer_model_path);

    return state;
}

fn deinit(allocator: std.mem.Allocator, raw_state: ?*anyopaque) void {
    const raw = raw_state orelse return;
    const state: *State = @ptrCast(@alignCast(raw));
    state.destroy(allocator);
}

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    switch (request.input) {
        .text => {},
        else => return error.InvalidInputPayload,
    }

    const state = stateFromHandle(handle) orelse return error.MissingProviderState;
    const Receipt = struct {
        status: []const u8,
        provider_key: []const u8,
        model_family: []const u8,
        model_id: []const u8,
        operation: []const u8,
        input_text: ?[]const u8,
        manifest_path: []const u8,
        tts_meta_path: []const u8,
        codec_meta_path: []const u8,
        tokenizer_model_path: []const u8,
        builtin_voice_count: usize,
        sample_rate: usize,
        channels: usize,
        num_quantizers: usize,
        output_contract: []const u8,
        message: []const u8,
    };

    const receipt = Receipt{
        .status = "runtime_backend_ready",
        .provider_key = handle.normalized.provider_key.name(),
        .model_family = handle.normalized.descriptor.family,
        .model_id = handle.normalized.descriptor.id,
        .operation = request.operation,
        .input_text = request.input.asString(),
        .manifest_path = state.manifest_path,
        .tts_meta_path = state.tts_meta_path,
        .codec_meta_path = state.codec_meta_path,
        .tokenizer_model_path = state.tokenizer_model_path,
        .builtin_voice_count = state.builtin_voice_count,
        .sample_rate = state.sample_rate,
        .channels = state.channels,
        .num_quantizers = state.num_quantizers,
        .output_contract = "audio_path",
        .message = "MOSS-TTS-Nano ONNX assets are now recognized by the unified runtime. Native zero-dependency TTS execution is not implemented yet, but this model family is no longer routed through generic fallback.",
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);

    return .{
        .origin = .shared_adapter,
        .note = .tts_model_ready,
        .output = .{ .json = try allocator.dupe(u8, out.written()) },
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}

fn readJsonFile(allocator: std.mem.Allocator, comptime T: type, path: []const u8) !T {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);
    return try std.json.parseFromSliceLeaky(T, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
}
