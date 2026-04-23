const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const bundle = @import("bundle/index.zig");
const frontend = @import("frontend/index.zig");
const inference = @import("inference/index.zig");
const tokenizer_mod = @import("tokenizer/index.zig");
const types = @import("../../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .moss_tts_nano_tts,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    manifest_path: []u8,
    tts_meta_path: []u8,
    codec_meta_path: []u8,
    tokenizer_model_path: []u8,
    manifest: bundle.ManifestSummary,
    tts: bundle.TtsSummary,
    codec: bundle.CodecConfig,
    tokenizer: ?tokenizer_mod.sentencepiece.Model = null,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        if (self.tokenizer) |*tokenizer| tokenizer.deinit();
        self.manifest.deinit();
        self.tts.deinit();
        self.codec.deinit();
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
    var loaded = try bundle.load(allocator, model.artifacts.model_dir) orelse return error.ModelManifestNotFound;
    defer loaded.deinit();
    var moved_manifest = loaded.manifest;
    loaded.manifest = .{};
    errdefer moved_manifest.deinit();
    var moved_tts = loaded.tts;
    loaded.tts = .{};
    errdefer moved_tts.deinit();
    var moved_codec = loaded.codec;
    loaded.codec = .{};
    errdefer moved_codec.deinit();

    const state = try allocator.create(State);
    errdefer allocator.destroy(state);
    state.* = .{
        .base = .{
            .provider_key = .moss_tts_nano_tts,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .manifest_path = try allocator.dupe(u8, loaded.paths.manifest_path),
        .tts_meta_path = try allocator.dupe(u8, loaded.paths.tts_meta_path),
        .codec_meta_path = try allocator.dupe(u8, loaded.paths.codec_meta_path),
        .tokenizer_model_path = try allocator.dupe(u8, loaded.paths.tokenizer_model_path),
        .manifest = moved_manifest,
        .tts = moved_tts,
        .codec = moved_codec,
        .tokenizer = tokenizer_mod.sentencepiece.loadFromFile(allocator, loaded.paths.tokenizer_model_path) catch null,
    };
    moved_manifest = .{};
    moved_tts = .{};
    moved_codec = .{};
    errdefer if (state.tokenizer) |*tokenizer| tokenizer.deinit();
    errdefer state.manifest.deinit();
    errdefer state.tts.deinit();
    errdefer state.codec.deinit();
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
    var prepared_text = try frontend.text.prepare(allocator, request.input.asString() orelse return error.MissingInputPayload, .{});
    defer prepared_text.deinit();
    const token_plan = try inference.planning.token.build(allocator, tokenizerFromState(state), prepared_text);
    defer token_plan.deinit();
    const request_plan = try inference.planning.request.build(allocator, state.manifest, token_plan);
    defer request_plan.deinit();

    const Receipt = struct {
        status: []const u8,
        provider_key: []const u8,
        model_family: []const u8,
        model_id: []const u8,
        operation: []const u8,
        input_text: ?[]const u8,
        normalized_text: []const u8,
        text_chunks: []const []const u8,
        chunk_estimated_tokens: []const usize,
        chunk_token_counts: []const usize,
        chunk_prefill_sequence_lengths: []const usize,
        chunk_request_row_widths: []const usize,
        chunk_prompt_audio_frame_counts: []const usize,
        chunk_count: usize,
        uses_estimated_token_budget: bool,
        tokenizer_loaded: bool,
        tokenizer_piece_count: usize,
        tokenizer_unk_id: ?usize,
        tokenizer_bos_id: ?usize,
        tokenizer_eos_id: ?usize,
        tokenizer_pad_id: ?usize,
        request_rows_ready: bool,
        tts_n_vq: usize,
        tts_vocab_size: usize,
        generation_max_new_frames: usize,
        generation_sample_mode: []const u8,
        default_voice: []const u8,
        manifest_path: []const u8,
        tts_meta_path: []const u8,
        codec_meta_path: []const u8,
        tokenizer_model_path: []const u8,
        builtin_voice_count: usize,
        has_generation_defaults: bool,
        has_model_files: bool,
        has_tts_model_info: bool,
        has_tts_session_options: bool,
        sample_rate: usize,
        channels: usize,
        num_quantizers: usize,
        downsample_rate: usize,
        graph_contract_ready: bool,
        tts_prefill_model_file: []const u8,
        tts_decode_step_model_file: []const u8,
        tts_local_cached_step_model_file: []const u8,
        tts_prefill_input_count: usize,
        tts_prefill_output_count: usize,
        tts_decode_input_count: usize,
        tts_decode_output_count: usize,
        tts_local_cached_input_count: usize,
        tts_local_cached_output_count: usize,
        codec_decode_full_model_file: []const u8,
        codec_decode_step_model_file: []const u8,
        codec_decode_input_count: usize,
        codec_decode_output_count: usize,
        codec_decode_step_input_count: usize,
        codec_decode_step_output_count: usize,
        codec_streaming_transformer_offset_count: usize,
        codec_streaming_attention_cache_count: usize,
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
        .normalized_text = prepared_text.normalized,
        .text_chunks = prepared_text.chunks,
        .chunk_estimated_tokens = prepared_text.estimated_tokens,
        .chunk_token_counts = token_plan.chunk_token_counts,
        .chunk_prefill_sequence_lengths = request_plan.prefill_sequence_lengths,
        .chunk_request_row_widths = request_plan.row_widths,
        .chunk_prompt_audio_frame_counts = request_plan.prompt_audio_frame_counts,
        .chunk_count = prepared_text.chunks.len,
        .uses_estimated_token_budget = token_plan.uses_estimated_token_budget,
        .tokenizer_loaded = token_plan.tokenizer_loaded,
        .tokenizer_piece_count = token_plan.tokenizer_summary.piece_count,
        .tokenizer_unk_id = token_plan.tokenizer_summary.unk_id,
        .tokenizer_bos_id = token_plan.tokenizer_summary.bos_id,
        .tokenizer_eos_id = token_plan.tokenizer_summary.eos_id,
        .tokenizer_pad_id = token_plan.tokenizer_summary.pad_id,
        .request_rows_ready = request_plan.ready,
        .tts_n_vq = state.manifest.tts_config.n_vq,
        .tts_vocab_size = state.manifest.tts_config.vocab_size,
        .generation_max_new_frames = state.manifest.generation_defaults.max_new_frames,
        .generation_sample_mode = state.manifest.generation_defaults.sample_mode,
        .default_voice = state.manifest.default_voice_name,
        .manifest_path = state.manifest_path,
        .tts_meta_path = state.tts_meta_path,
        .codec_meta_path = state.codec_meta_path,
        .tokenizer_model_path = state.tokenizer_model_path,
        .builtin_voice_count = state.manifest.builtin_voice_count,
        .has_generation_defaults = state.manifest.has_generation_defaults,
        .has_model_files = state.manifest.has_model_files,
        .has_tts_model_info = state.tts.has_model_info,
        .has_tts_session_options = state.tts.has_session_options,
        .sample_rate = state.codec.sample_rate,
        .channels = state.codec.channels,
        .num_quantizers = state.codec.num_quantizers,
        .downsample_rate = state.codec.downsample_rate,
        .graph_contract_ready = graphContractReady(state),
        .tts_prefill_model_file = state.tts.files.prefill,
        .tts_decode_step_model_file = state.tts.files.decode_step,
        .tts_local_cached_step_model_file = state.tts.files.local_cached_step,
        .tts_prefill_input_count = state.tts.onnx.prefill_input_names.len,
        .tts_prefill_output_count = state.tts.onnx.prefill_output_names.len,
        .tts_decode_input_count = state.tts.onnx.decode_input_names.len,
        .tts_decode_output_count = state.tts.onnx.decode_output_names.len,
        .tts_local_cached_input_count = state.tts.onnx.local_cached_input_names.len,
        .tts_local_cached_output_count = state.tts.onnx.local_cached_output_names.len,
        .codec_decode_full_model_file = state.codec.files.decode_full,
        .codec_decode_step_model_file = state.codec.files.decode_step,
        .codec_decode_input_count = state.codec.onnx.decode_input_names.len,
        .codec_decode_output_count = state.codec.onnx.decode_output_names.len,
        .codec_decode_step_input_count = state.codec.onnx.decode_step_input_names.len,
        .codec_decode_step_output_count = state.codec.onnx.decode_step_output_names.len,
        .codec_streaming_transformer_offset_count = state.codec.streaming_transformer_offset_count,
        .codec_streaming_attention_cache_count = state.codec.streaming_attention_cache_count,
        .output_contract = "audio_path",
        .message = "MOSS-TTS-Nano assets, tokenizer, and voice-clone prefill request rows are now prepared by the unified runtime. Native zero-dependency acoustic and codec graph execution are not implemented yet.",
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

fn tokenizerFromState(state: *const State) ?*const tokenizer_mod.sentencepiece.Model {
    if (state.tokenizer) |*tokenizer| return tokenizer;
    return null;
}

fn graphContractReady(state: *const State) bool {
    return state.tts.files.prefill.len != 0 and
        state.tts.files.decode_step.len != 0 and
        state.tts.files.local_cached_step.len != 0 and
        state.tts.onnx.prefill_input_names.len != 0 and
        state.tts.onnx.prefill_output_names.len != 0 and
        state.tts.onnx.decode_input_names.len != 0 and
        state.tts.onnx.decode_output_names.len != 0 and
        state.codec.files.decode_full.len != 0 and
        state.codec.files.decode_step.len != 0 and
        state.codec.onnx.decode_input_names.len != 0 and
        state.codec.onnx.decode_output_names.len != 0 and
        state.codec.onnx.decode_step_input_names.len != 0 and
        state.codec.onnx.decode_step_output_names.len != 0;
}
