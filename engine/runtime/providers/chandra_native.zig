const std = @import("std");
const imaging = @import("Pixio");
const task = @import("../../core/task.zig");
const preprocess = @import("chandra_preprocess.zig");
const store = @import("chandra_store.zig");
const vision = @import("chandra_vision.zig");
const weights = @import("chandra_weights.zig");
const decoder_runtime = @import("../text/decoder_runtime.zig");
const decoder_family = @import("../text/decoder_family.zig");
const decoder_types = @import("../text/decoder_types.zig");
const text_backend_scheme = @import("../text/backend_scheme.zig");
const kv_cache = @import("../text/kv_cache.zig");
const streaming = @import("../text/streaming.zig");

const io = std.Options.debug_io;

pub const TextConfig = struct {
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_parameters: ?std.json.Value = null,
};

pub const VisionConfig = struct {
    model_type: []const u8,
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    out_hidden_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
};

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,
    image_token_id: usize,
    video_token_id: ?usize = null,
    vision_start_token_id: usize,
    vision_end_token_id: usize,
    text_config: TextConfig,
    vision_config: VisionConfig,

    pub fn isSupportedChandraShape(self: Config) bool {
        return std.mem.eql(u8, self.model_type, "qwen3_5") and
            std.mem.eql(u8, self.text_config.model_type, "qwen3_5_text") and
            self.text_config.hidden_size == self.vision_config.out_hidden_size;
    }
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: Config,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const Context = struct {
    operation: []const u8,
    model_path: []const u8,
    input_path: []const u8,
    execution: task.ExecutionMode,
    max_output_tokens: ?usize = null,
};

pub const Readiness = struct {
    has_config: bool,
    has_tokenizer: bool,
    has_weights: bool,
    has_supported_config: bool,
    has_visual_encoder: bool,
    has_patch_embedding_weight: bool,
    has_multimodal_projector: bool,
    has_document_preprocessor: bool,
    has_image_processor_config: bool,
    text_tensor_count: usize = 0,
    vision_tensor_count: usize = 0,
    projector_tensor_count: usize = 0,
    output_tensor_count: usize = 0,
};

pub const LoadedModel = struct {
    readiness: Readiness,
    parsed_config: ?*const ParsedConfig = null,
    image_processor: ?*const preprocess.ParsedImageProcessorConfig = null,
    tensor_store: ?*const store.ChandraStore = null,
};

const PreprocessSummary = struct {
    image_width: usize,
    image_height: usize,
    resized_width: usize,
    resized_height: usize,
    patch_token_count: usize,
    visual_token_count: usize,
    vision_block_depth: usize = 0,
    patch_embedding_dim: ?usize = null,
    patch_embedding_executed: bool = false,
    visual_position_dim: ?usize = null,
    visual_position_embedding_executed: bool = false,
    visual_attention_dim: ?usize = null,
    visual_block_attention_executed: bool = false,
    visual_attention_blocks_executed: usize = 0,
    visual_block_dim: ?usize = null,
    visual_block_mlp_executed: bool = false,
    visual_mlp_blocks_executed: usize = 0,
    visual_token_dim: ?usize = null,
    visual_merger_executed: bool = false,
    text_prompt_token_count: usize = 0,
    text_prefill_token_count: usize = 0,
    text_prefill_executed: bool = false,
    decoder_rope_position_mode: ?[]const u8 = null,
    decoder_mrope_sections: [4]u32 = .{ 0, 0, 0, 0 },
    decoder_logits_dim: ?usize = null,
    decoder_next_token_id: ?usize = null,
    text_decode_executed: bool = false,
    decoded_token_count: usize = 0,
    generated_output: ?[]u8 = null,

    pub fn deinit(self: *PreprocessSummary, allocator: std.mem.Allocator) void {
        if (self.generated_output) |value| allocator.free(value);
        self.* = undefined;
    }
};

const MultimodalPositionPlan = struct {
    vision_start_position: usize,
    visual_start_position: usize,
    vision_end_position: usize,
    prompt_start_position: usize,
    generation_start_position: usize,
    total_prefill_tokens: usize,

    fn init(
        rope_position_mode: decoder_types.RopePositionMode,
        visual_token_count: usize,
        visual_grid_time: usize,
        visual_grid_width: usize,
        visual_grid_height: usize,
        prompt_token_count: usize,
    ) MultimodalPositionPlan {
        const vision_start_position: usize = 0;
        const visual_start_position = vision_start_position + 1;
        const visual_max_position = maxVisualPosition(
            rope_position_mode,
            visual_start_position,
            visual_token_count,
            visual_grid_time,
            visual_grid_width,
            visual_grid_height,
        );
        const vision_end_position = visual_max_position + 1;
        const prompt_start_position = vision_end_position + 1;
        const generation_start_position = prompt_start_position + prompt_token_count;
        return .{
            .vision_start_position = vision_start_position,
            .visual_start_position = visual_start_position,
            .vision_end_position = vision_end_position,
            .prompt_start_position = prompt_start_position,
            .generation_start_position = generation_start_position,
            .total_prefill_tokens = 1 + visual_token_count + 1 + prompt_token_count,
        };
    }
};

fn visualTokenPosition(
    rope_position_mode: decoder_types.RopePositionMode,
    base_position: usize,
    token_index: usize,
    grid_time: usize,
    grid_height: usize,
    grid_width: usize,
) decoder_types.TokenPosition {
    const effective_grid_time = @max(@as(usize, 1), grid_time);
    const effective_grid_height = @max(@as(usize, 1), grid_height);
    const effective_grid_width = @max(@as(usize, 1), grid_width);
    if (rope_position_mode == .scalar) {
        return decoder_types.TokenPosition.scalarPosition(base_position + token_index);
    }

    const tokens_per_frame = effective_grid_height * effective_grid_width;
    const t = token_index / tokens_per_frame;
    const frame_offset = token_index % tokens_per_frame;
    const y = frame_offset / effective_grid_width;
    const x = frame_offset % effective_grid_width;
    const scalar_position = base_position + @max(y, x);
    return decoder_types.TokenPosition.mropePosition(.{
        base_position + @min(t, effective_grid_time - 1),
        base_position + y,
        base_position + x,
        scalar_position,
    });
}

fn textTokenPosition(
    rope_position_mode: decoder_types.RopePositionMode,
    scalar_position: usize,
) decoder_types.TokenPosition {
    if (rope_position_mode == .scalar) {
        return decoder_types.TokenPosition.scalarPosition(scalar_position);
    }
    return decoder_types.TokenPosition.mropePosition(.{ scalar_position, scalar_position, scalar_position, scalar_position });
}

fn maxVisualPosition(
    rope_position_mode: decoder_types.RopePositionMode,
    visual_start_position: usize,
    visual_token_count: usize,
    visual_grid_time: usize,
    visual_grid_width: usize,
    visual_grid_height: usize,
) usize {
    if (visual_token_count == 0) return visual_start_position;
    if (rope_position_mode != .mrope or visual_grid_width == 0 or visual_grid_height == 0) {
        return visual_start_position + visual_token_count - 1;
    }

    const max_time_index = @max(@as(usize, 1), visual_grid_time) - 1;
    const max_height_index = visual_grid_height - 1;
    const max_width_index = visual_grid_width - 1;
    return visual_start_position + @max(max_time_index, @max(max_height_index, max_width_index));
}

pub fn execute(allocator: std.mem.Allocator, context: Context) ![]u8 {
    return try executeWithLoadedModel(allocator, context, .{
        .readiness = inspect(context.model_path),
    });
}

pub fn executeWithLoadedModel(
    allocator: std.mem.Allocator,
    context: Context,
    loaded_model: LoadedModel,
) ![]u8 {
    if (context.execution != .sync) return error.UnsupportedExecutionMode;
    const readiness = loaded_model.readiness;
    var summary: ?PreprocessSummary = null;
    defer if (summary) |*value| value.deinit(allocator);

    if (readiness.has_image_processor_config and isSupportedInputPath(context.input_path)) {
        var owned_image_processor: ?preprocess.ParsedImageProcessorConfig = null;
        defer if (owned_image_processor) |*value| value.deinit();
        var image_processor = loaded_model.image_processor;
        if (image_processor == null) {
            owned_image_processor = try preprocess.loadImageProcessorConfig(allocator, context.model_path);
            if (owned_image_processor) |*value| image_processor = value;
        }
        const image_processor_config = image_processor.?.value;

        var prepared = try loadPreparedInputFromPath(allocator, context.input_path, image_processor_config);
        defer prepared.deinit();

        var owned_parsed_config: ?ParsedConfig = null;
        defer if (owned_parsed_config) |*config| config.deinit();
        var parsed_config = loaded_model.parsed_config;
        if (parsed_config == null) {
            const config_path = std.fs.path.join(allocator, &.{ context.model_path, "config.json" }) catch null;
            defer if (config_path) |path| allocator.free(path);
            if (config_path) |path| {
                owned_parsed_config = loadConfigFromFile(allocator, path) catch null;
                if (owned_parsed_config) |*config| parsed_config = config;
            }
        }

        const vision_block_depth = if (parsed_config) |config|
            config.value.vision_config.depth
        else
            0;
        const vision_num_heads = if (parsed_config) |config|
            @max(@as(usize, 1), config.value.vision_config.num_heads)
        else
            1;

        var patch_embedding_dim: ?usize = null;
        var patch_embedding_executed = false;
        var visual_position_dim: ?usize = null;
        var visual_position_embedding_executed = false;
        var visual_attention_dim: ?usize = null;
        var visual_block_attention_executed = false;
        var visual_attention_blocks_executed: usize = 0;
        var visual_block_dim: ?usize = null;
        var visual_block_mlp_executed = false;
        var visual_mlp_blocks_executed: usize = 0;
        var visual_token_dim: ?usize = null;
        var visual_merger_executed = false;
        var text_prompt_token_count: usize = 0;
        var text_prefill_token_count: usize = 0;
        var text_prefill_executed = false;
        var decoder_rope_position_mode: ?[]const u8 = null;
        var decoder_mrope_sections: [4]u32 = .{ 0, 0, 0, 0 };
        var decoder_logits_dim: ?usize = null;
        var decoder_next_token_id: ?usize = null;
        var text_decode_executed = false;
        var decoded_token_count: usize = 0;
        var generated_output: ?[]u8 = null;
        if (readiness.has_patch_embedding_weight) {
            var owned_tensor_store: ?store.ChandraStore = null;
            defer if (owned_tensor_store) |*opened| opened.deinit();
            var tensor_store = loaded_model.tensor_store;
            if (tensor_store == null) {
                owned_tensor_store = store.ChandraStore.open(allocator, context.model_path) catch null;
                if (owned_tensor_store) |*opened| tensor_store = opened;
            }
            if (tensor_store) |opened| {
                var patch_weights = opened.loadPatchEmbeddingWeights(allocator) catch null;
                if (patch_weights) |*loaded| {
                    defer loaded.deinit();
                    var embeddings = vision.patchEmbedImage(allocator, &prepared, loaded.weights) catch null;
                    if (embeddings) |*value| {
                        patch_embedding_dim = value.embedding_dim;
                        patch_embedding_executed = true;

                        var merged_input = value.*;
                        var current_owned: ?vision.PatchEmbeddings = null;
                        defer if (current_owned) |*owned| owned.deinit();

                        var pos_weights = opened.loadVisualPositionEmbeddings(allocator, value.embedding_dim) catch null;
                        if (pos_weights) |*pos| {
                            defer pos.deinit();
                            const positioned = vision.applyPositionEmbeddings(allocator, value.*, pos.weights) catch null;
                            if (positioned) |token_features| {
                                current_owned = token_features;
                                merged_input = current_owned.?;
                                visual_position_dim = merged_input.embedding_dim;
                                visual_position_embedding_executed = true;
                            }
                        }

                        for (0..vision_block_depth) |block_index| {
                            const block_attention = blk: {
                                var attn = opened.loadVisionBlockAttentionWeights(allocator, block_index, vision_num_heads) catch break;
                                defer attn.deinit();

                                const next = vision.applyVisionBlockAttention(allocator, merged_input, attn.weights, 1e-5) catch break;
                                break :blk next;
                            };
                            if (current_owned) |*owned| owned.deinit();
                            current_owned = block_attention;
                            merged_input = current_owned.?;
                            visual_attention_dim = merged_input.embedding_dim;
                            visual_block_attention_executed = true;
                            visual_attention_blocks_executed = block_index + 1;

                            const block_mlp = blk: {
                                var mlp = opened.loadVisionBlockMlpWeights(allocator, block_index) catch break;
                                defer mlp.deinit();

                                const next = vision.applyVisionBlockMlp(allocator, merged_input, mlp.weights, 1e-5) catch break;
                                break :blk next;
                            };
                            if (current_owned) |*owned| owned.deinit();
                            current_owned = block_mlp;
                            merged_input = current_owned.?;
                            visual_block_dim = merged_input.embedding_dim;
                            visual_block_mlp_executed = true;
                            visual_mlp_blocks_executed = block_index + 1;
                        }

                        var grouped = vision.mergeSpatialPatches(allocator, merged_input, image_processor_config.merge_size) catch null;
                        if (grouped) |*merged| {
                            defer merged.deinit();
                            var merger_weights = opened.loadVisualMergerWeights(allocator) catch null;
                            if (merger_weights) |*projector| {
                                defer projector.deinit();
                                var visual_tokens = vision.applyVisualMerger(allocator, merged.*, projector.weights) catch null;
                                if (visual_tokens) |*tokens| {
                                    visual_token_dim = tokens.embedding_dim;
                                    visual_merger_executed = true;

                                    if (parsed_config) |config| {
                                        var text_runtime = decoder_runtime.initRuntime(
                                            allocator,
                                            context.model_path,
                                            text_backend_scheme.Scheme.auto,
                                            1,
                                        ) catch null;
                                        if (text_runtime) |*runtime| {
                                            defer runtime.deinit();
                                            decoder_rope_position_mode = runtime.cfg.rope_position_mode.name();
                                            decoder_mrope_sections = runtime.cfg.mrope_sections;

                                            var prompt_token_ids: ?[]usize = null;
                                            defer if (prompt_token_ids) |ids| allocator.free(ids);
                                            var tokenizer = decoder_family.loadTokenizerFromModelDir(
                                                allocator,
                                                runtime.cfg.architecture,
                                                context.model_path,
                                            ) catch null;
                                            defer if (tokenizer) |*loaded_tokenizer| loaded_tokenizer.deinit();
                                            if (tokenizer) |*loaded_tokenizer| {
                                                const prompt_text = decoder_family.renderSingleUserPromptAlloc(
                                                    allocator,
                                                    runtime.cfg.architecture,
                                                    instructionForOperation(context.operation),
                                                    .disabled,
                                                ) catch null;
                                                if (prompt_text) |text| {
                                                    defer allocator.free(text);
                                                    const encoded = loaded_tokenizer.encodeAlloc(allocator, text) catch null;
                                                    if (encoded) |ids_u32| {
                                                        defer allocator.free(ids_u32);
                                                        const ids = allocator.alloc(usize, ids_u32.len) catch null;
                                                        if (ids) |owned_ids| {
                                                            for (ids_u32, 0..) |token_id, index| {
                                                                owned_ids[index] = token_id;
                                                            }
                                                            prompt_token_ids = owned_ids;
                                                            text_prompt_token_count = owned_ids.len;
                                                        }
                                                    }
                                                }
                                            }

                                            const position_plan = MultimodalPositionPlan.init(
                                                runtime.cfg.rope_position_mode,
                                                tokens.token_count,
                                                tokens.grid_time,
                                                tokens.grid_width,
                                                tokens.grid_height,
                                                text_prompt_token_count,
                                            );
                                            const resolved_kv_cache_scheme = kv_cache.resolveScheme(.auto, runtime.backendName());
                                            var cache = kv_cache.ModelCache.initWithLayout(
                                                allocator,
                                                runtime.cfg.num_hidden_layers,
                                                position_plan.total_prefill_tokens + (context.max_output_tokens orelse 64),
                                                runtime.cfg.num_key_value_heads,
                                                runtime.cfg.head_dim,
                                                resolved_kv_cache_scheme,
                                                kv_cache.default_q8_layout,
                                            ) catch null;
                                            if (cache) |*model_cache| {
                                                defer model_cache.deinit();
                                                var workspace = runtime.initWorkspace(position_plan.total_prefill_tokens + (context.max_output_tokens orelse 64)) catch null;
                                                if (workspace) |*decoder_workspace| {
                                                    defer decoder_workspace.deinit();

                                                    const current_logits = runtime.forwardTokenIdWithPosition(
                                                        decoder_workspace,
                                                        model_cache,
                                                        config.value.vision_start_token_id,
                                                        textTokenPosition(runtime.cfg.rope_position_mode, position_plan.vision_start_position),
                                                    ) catch null;
                                                    if (current_logits) |start_logits| {
                                                        var latest_logits = start_logits;
                                                        const visual_positions = allocMultimodalVisualPositions(
                                                            allocator,
                                                            runtime.cfg.rope_position_mode,
                                                            position_plan.visual_start_position,
                                                            tokens.token_count,
                                                            tokens.grid_time,
                                                            tokens.grid_height,
                                                            tokens.grid_width,
                                                        ) catch null;
                                                        defer if (visual_positions) |positions| allocator.free(positions);
                                                        if (visual_positions) |positions| {
                                                            latest_logits = runtime.prefillEmbeddingsWithPositions(
                                                                decoder_workspace,
                                                                model_cache,
                                                                tokens.data,
                                                                tokens.token_count,
                                                                positions,
                                                            ) catch latest_logits;
                                                        }
                                                        latest_logits = runtime.forwardTokenIdWithPosition(
                                                            decoder_workspace,
                                                            model_cache,
                                                            config.value.vision_end_token_id,
                                                            textTokenPosition(runtime.cfg.rope_position_mode, position_plan.vision_end_position),
                                                        ) catch latest_logits;
                                                        if (prompt_token_ids) |ids| {
                                                            const prompt_positions = allocMultimodalTextPositions(
                                                                allocator,
                                                                runtime.cfg.rope_position_mode,
                                                                position_plan.prompt_start_position,
                                                                ids.len,
                                                            ) catch null;
                                                            defer if (prompt_positions) |positions| allocator.free(positions);
                                                            if (prompt_positions) |positions| {
                                                                latest_logits = runtime.prefillTokenIdsWithPositions(
                                                                    decoder_workspace,
                                                                    model_cache,
                                                                    ids,
                                                                    positions,
                                                                ) catch latest_logits;
                                                            }
                                                        }

                                                        text_prefill_token_count = position_plan.total_prefill_tokens;
                                                        text_prefill_executed = true;
                                                        decoder_logits_dim = latest_logits.len;
                                                        decoder_next_token_id = decoder_family.argMaxLogit(latest_logits) catch null;

                                                        if (tokenizer) |*loaded_tokenizer| {
                                                            var generated_ids = std.ArrayListUnmanaged(u32).empty;
                                                            defer generated_ids.deinit(allocator);

                                                            var step: usize = 0;
                                                            const max_output_tokens = context.max_output_tokens orelse 64;
                                                            while (step < max_output_tokens) : (step += 1) {
                                                                const next_token = decoder_family.argMaxLogit(latest_logits) catch break;
                                                                if (decoder_family.isEosToken(runtime.cfg.architecture, next_token)) break;
                                                                const next_token_u32 = std.math.cast(u32, next_token) orelse break;
                                                                try generated_ids.append(allocator, next_token_u32);
                                                                latest_logits = runtime.forwardTokenIdWithPosition(
                                                                    decoder_workspace,
                                                                    model_cache,
                                                                    next_token,
                                                                    textTokenPosition(runtime.cfg.rope_position_mode, position_plan.generation_start_position + step),
                                                                ) catch break;
                                                            }

                                                            decoder_logits_dim = latest_logits.len;
                                                            decoder_next_token_id = decoder_family.argMaxLogit(latest_logits) catch decoder_next_token_id;
                                                            decoded_token_count = generated_ids.items.len;
                                                            text_decode_executed = generated_ids.items.len != 0;

                                                            if (generated_ids.items.len != 0) {
                                                                const decoded = loaded_tokenizer.decodeAlloc(allocator, generated_ids.items) catch null;
                                                                if (decoded) |text| {
                                                                    const effective_stops = decoder_family.effectiveStopSequencesAlloc(
                                                                        allocator,
                                                                        runtime.cfg.architecture,
                                                                        &.{},
                                                                    ) catch null;
                                                                    if (effective_stops) |stops| {
                                                                        defer allocator.free(stops);
                                                                        const analysis = streaming.analyzeGeneratedText(text, stops);
                                                                        generated_output = try allocator.dupe(u8, text[0..analysis.response_len]);
                                                                    } else {
                                                                        generated_output = try allocator.dupe(u8, text);
                                                                    }
                                                                    allocator.free(text);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    tokens.deinit();
                                }
                            }
                        }

                        value.deinit();
                    }
                }
            }
        }

        summary = .{
            .image_width = prepared.grid.input_width,
            .image_height = prepared.grid.input_height,
            .resized_width = prepared.grid.resized_width,
            .resized_height = prepared.grid.resized_height,
            .patch_token_count = prepared.grid.patch_token_count,
            .visual_token_count = prepared.grid.token_count,
            .vision_block_depth = vision_block_depth,
            .patch_embedding_dim = patch_embedding_dim,
            .patch_embedding_executed = patch_embedding_executed,
            .visual_position_dim = visual_position_dim,
            .visual_position_embedding_executed = visual_position_embedding_executed,
            .visual_attention_dim = visual_attention_dim,
            .visual_block_attention_executed = visual_block_attention_executed,
            .visual_attention_blocks_executed = visual_attention_blocks_executed,
            .visual_block_dim = visual_block_dim,
            .visual_block_mlp_executed = visual_block_mlp_executed,
            .visual_mlp_blocks_executed = visual_mlp_blocks_executed,
            .visual_token_dim = visual_token_dim,
            .visual_merger_executed = visual_merger_executed,
            .text_prompt_token_count = text_prompt_token_count,
            .text_prefill_token_count = text_prefill_token_count,
            .text_prefill_executed = text_prefill_executed,
            .decoder_rope_position_mode = decoder_rope_position_mode,
            .decoder_mrope_sections = decoder_mrope_sections,
            .decoder_logits_dim = decoder_logits_dim,
            .decoder_next_token_id = decoder_next_token_id,
            .text_decode_executed = text_decode_executed,
            .decoded_token_count = decoded_token_count,
            .generated_output = generated_output,
        };
    }

    return try buildIncompleteOutputJson(allocator, context, readiness, summary);
}

pub fn inspect(model_path: []const u8) Readiness {
    var supported_config = false;
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const config_path = std.fs.path.join(allocator, &.{ model_path, "config.json" }) catch null;
    defer if (config_path) |path| allocator.free(path);
    if (config_path) |path| {
        var parsed = loadConfigFromFile(allocator, path) catch null;
        if (parsed) |*config| {
            supported_config = config.value.isSupportedChandraShape();
            config.deinit();
        }
    }

    var weight_counts: weights.GroupCounts = .{};
    var has_weights = false;
    var has_patch_embedding_weight = false;
    var manifest = weights.loadManifest(allocator, model_path) catch null;
    if (manifest) |*loaded| {
        has_weights = loaded.len() != 0;
        weight_counts = loaded.counts;
        has_patch_embedding_weight = loaded.findPatchEmbeddingWeight() != null;
        loaded.deinit();
    }

    var has_image_processor_config = false;
    var image_processor = preprocess.loadImageProcessorConfig(allocator, model_path) catch null;
    if (image_processor) |*config| {
        has_image_processor_config = config.value.patch_size == 16 and config.value.merge_size != 0;
        config.deinit();
    }

    return .{
        .has_config = hasFile(model_path, "config.json"),
        .has_tokenizer = hasAnyFile(model_path, &.{ "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt" }),
        .has_weights = has_weights,
        .has_supported_config = supported_config,
        .has_visual_encoder = weight_counts.vision != 0,
        .has_patch_embedding_weight = has_patch_embedding_weight,
        .has_multimodal_projector = weight_counts.projector != 0,
        .has_document_preprocessor = has_image_processor_config,
        .has_image_processor_config = has_image_processor_config,
        .text_tensor_count = weight_counts.text,
        .vision_tensor_count = weight_counts.vision,
        .projector_tensor_count = weight_counts.projector,
        .output_tensor_count = weight_counts.output,
    };
}

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(2 * 1024 * 1024));
    const config = try std.json.parseFromSliceLeaky(Config, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{
        .arena = arena,
        .value = config,
    };
}

fn buildIncompleteOutputJson(
    allocator: std.mem.Allocator,
    context: Context,
    readiness: Readiness,
    preprocess_summary: ?PreprocessSummary,
) ![]u8 {
    const ReadinessReceipt = struct {
        has_config: bool,
        has_tokenizer: bool,
        has_weights: bool,
        has_supported_config: bool,
        has_visual_encoder: bool,
        has_patch_embedding_weight: bool,
        has_multimodal_projector: bool,
        has_document_preprocessor: bool,
        has_image_processor_config: bool,
        text_tensor_count: usize,
        vision_tensor_count: usize,
        projector_tensor_count: usize,
        output_tensor_count: usize,
    };

    const Receipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: []const u8,
        backend: []const u8,
        method: []const u8,
        requested_output: []const u8,
        native_stage: []const u8,
        content: ?[]const u8,
        markdown: ?[]const u8,
        html: ?[]const u8,
        json_output: ?[]const u8,
        page_count: ?usize,
        total_token_count: ?usize,
        loaded_tensors: ?usize,
        image_width: ?usize,
        image_height: ?usize,
        resized_width: ?usize,
        resized_height: ?usize,
        patch_token_count: ?usize,
        visual_token_count: ?usize,
        vision_block_depth: ?usize,
        patch_embedding_dim: ?usize,
        patch_embedding_executed: bool,
        visual_position_dim: ?usize,
        visual_position_embedding_executed: bool,
        visual_attention_dim: ?usize,
        visual_block_attention_executed: bool,
        visual_attention_blocks_executed: usize,
        visual_block_dim: ?usize,
        visual_block_mlp_executed: bool,
        visual_mlp_blocks_executed: usize,
        visual_token_dim: ?usize,
        visual_merger_executed: bool,
        text_prompt_token_count: usize,
        text_prefill_token_count: usize,
        text_prefill_executed: bool,
        decoder_rope_position_mode: ?[]const u8,
        decoder_mrope_sections: [4]u32,
        decoder_logits_dim: ?usize,
        decoder_next_token_id: ?usize,
        text_decode_executed: bool,
        decoded_token_count: usize,
        error_message: []const u8,
        readiness: ReadinessReceipt,
    };

    const receipt = Receipt{
        .status = if (preprocess_summary) |summary|
            if (summary.text_decode_executed and summary.generated_output != null)
                "ocr_native_text_decoded_partial"
            else
                "ocr_native_backend_incomplete"
        else
            "ocr_native_backend_incomplete",
        .operation = context.operation,
        .model_family = "chandra",
        .model_path = context.model_path,
        .input_path = context.input_path,
        .backend = "kinetix_native",
        .method = "native",
        .requested_output = requestedOutput(context.operation),
        .native_stage = if (preprocess_summary) |summary|
            if (summary.text_decode_executed)
                "text_decode"
            else if (summary.text_prefill_executed)
                "text_prefill"
            else if (summary.visual_merger_executed)
                "visual_merger"
            else if (summary.visual_position_embedding_executed and !summary.visual_block_attention_executed)
                "visual_position_embedding"
            else if (summary.visual_block_attention_executed and summary.visual_block_mlp_executed)
                "visual_block_mlp"
            else if (summary.visual_block_attention_executed)
                "visual_block_attention"
            else if (summary.visual_block_mlp_executed)
                "visual_block_mlp"
            else if (summary.patch_embedding_executed)
                "patch_embedding"
            else
                "image_preprocessing"
        else
            "model_loading",
        .content = if (preprocess_summary) |summary| summary.generated_output else null,
        .markdown = if (preprocess_summary) |summary|
            if (std.mem.eql(u8, context.operation, "render-markdown"))
                summary.generated_output
            else
                null
        else
            null,
        .html = if (preprocess_summary) |summary|
            if (std.mem.eql(u8, context.operation, "render-html"))
                summary.generated_output
            else
                null
        else
            null,
        .json_output = if (preprocess_summary) |summary|
            if (std.mem.eql(u8, context.operation, "render-json"))
                summary.generated_output
            else
                null
        else
            null,
        .page_count = if (preprocess_summary != null) 1 else null,
        .total_token_count = if (preprocess_summary) |summary|
            summary.text_prefill_token_count + summary.decoded_token_count
        else
            null,
        .loaded_tensors = null,
        .image_width = if (preprocess_summary) |summary| summary.image_width else null,
        .image_height = if (preprocess_summary) |summary| summary.image_height else null,
        .resized_width = if (preprocess_summary) |summary| summary.resized_width else null,
        .resized_height = if (preprocess_summary) |summary| summary.resized_height else null,
        .patch_token_count = if (preprocess_summary) |summary| summary.patch_token_count else null,
        .visual_token_count = if (preprocess_summary) |summary| summary.visual_token_count else null,
        .vision_block_depth = if (preprocess_summary) |summary| summary.vision_block_depth else null,
        .patch_embedding_dim = if (preprocess_summary) |summary| summary.patch_embedding_dim else null,
        .patch_embedding_executed = if (preprocess_summary) |summary| summary.patch_embedding_executed else false,
        .visual_position_dim = if (preprocess_summary) |summary| summary.visual_position_dim else null,
        .visual_position_embedding_executed = if (preprocess_summary) |summary| summary.visual_position_embedding_executed else false,
        .visual_attention_dim = if (preprocess_summary) |summary| summary.visual_attention_dim else null,
        .visual_block_attention_executed = if (preprocess_summary) |summary| summary.visual_block_attention_executed else false,
        .visual_attention_blocks_executed = if (preprocess_summary) |summary| summary.visual_attention_blocks_executed else 0,
        .visual_block_dim = if (preprocess_summary) |summary| summary.visual_block_dim else null,
        .visual_block_mlp_executed = if (preprocess_summary) |summary| summary.visual_block_mlp_executed else false,
        .visual_mlp_blocks_executed = if (preprocess_summary) |summary| summary.visual_mlp_blocks_executed else 0,
        .visual_token_dim = if (preprocess_summary) |summary| summary.visual_token_dim else null,
        .visual_merger_executed = if (preprocess_summary) |summary| summary.visual_merger_executed else false,
        .text_prompt_token_count = if (preprocess_summary) |summary| summary.text_prompt_token_count else 0,
        .text_prefill_token_count = if (preprocess_summary) |summary| summary.text_prefill_token_count else 0,
        .text_prefill_executed = if (preprocess_summary) |summary| summary.text_prefill_executed else false,
        .decoder_rope_position_mode = if (preprocess_summary) |summary| summary.decoder_rope_position_mode else null,
        .decoder_mrope_sections = if (preprocess_summary) |summary| summary.decoder_mrope_sections else .{ 0, 0, 0, 0 },
        .decoder_logits_dim = if (preprocess_summary) |summary| summary.decoder_logits_dim else null,
        .decoder_next_token_id = if (preprocess_summary) |summary| summary.decoder_next_token_id else null,
        .text_decode_executed = if (preprocess_summary) |summary| summary.text_decode_executed else false,
        .decoded_token_count = if (preprocess_summary) |summary| summary.decoded_token_count else 0,
        .error_message = if (preprocess_summary) |summary|
            if (summary.text_decode_executed and summary.generated_output != null)
                "Chandra native OCR can now execute multimodal prefill, M-RoPE-aware decoding, and partial text generation, but full OCR quality and broader multimodal parity are still in progress."
            else
                "Chandra native inference is not complete yet; model config, weight manifest, and preprocessing readiness are available."
        else
            "Chandra native inference is not complete yet; model config, weight manifest, and preprocessing readiness are available.",
        .readiness = .{
            .has_config = readiness.has_config,
            .has_tokenizer = readiness.has_tokenizer,
            .has_weights = readiness.has_weights,
            .has_supported_config = readiness.has_supported_config,
            .has_visual_encoder = readiness.has_visual_encoder,
            .has_patch_embedding_weight = readiness.has_patch_embedding_weight,
            .has_multimodal_projector = readiness.has_multimodal_projector,
            .has_document_preprocessor = readiness.has_document_preprocessor,
            .has_image_processor_config = readiness.has_image_processor_config,
            .text_tensor_count = readiness.text_tensor_count,
            .vision_tensor_count = readiness.vision_tensor_count,
            .projector_tensor_count = readiness.projector_tensor_count,
            .output_tensor_count = readiness.output_tensor_count,
        },
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn requestedOutput(operation: []const u8) []const u8 {
    if (std.mem.eql(u8, operation, "render-markdown")) return "markdown";
    if (std.mem.eql(u8, operation, "render-html")) return "html";
    if (std.mem.eql(u8, operation, "render-json")) return "json";
    return "markdown";
}

fn instructionForOperation(operation: []const u8) []const u8 {
    if (std.mem.eql(u8, operation, "render-markdown")) return "Read the document image and transcribe it as markdown.";
    if (std.mem.eql(u8, operation, "render-html")) return "Read the document image and transcribe it as semantic HTML.";
    if (std.mem.eql(u8, operation, "render-json")) return "Read the document image and transcribe it as structured JSON.";
    return "Read the document image and transcribe all visible text.";
}

fn allocMultimodalTextPositions(
    allocator: std.mem.Allocator,
    rope_position_mode: decoder_types.RopePositionMode,
    start: usize,
    count: usize,
) ![]decoder_types.TokenPosition {
    const positions = try allocator.alloc(decoder_types.TokenPosition, count);
    for (positions, 0..) |*position, index| {
        position.* = textTokenPosition(rope_position_mode, start + index);
    }
    return positions;
}

fn allocMultimodalVisualPositions(
    allocator: std.mem.Allocator,
    rope_position_mode: decoder_types.RopePositionMode,
    start: usize,
    count: usize,
    grid_time: usize,
    grid_height: usize,
    grid_width: usize,
) ![]decoder_types.TokenPosition {
    const positions = try allocator.alloc(decoder_types.TokenPosition, count);
    for (positions, 0..) |*position, index| {
        position.* = visualTokenPosition(
            rope_position_mode,
            start,
            index,
            grid_time,
            grid_height,
            grid_width,
        );
    }
    return positions;
}

fn isSupportedInputPath(path: []const u8) bool {
    return isRasterImagePath(path) or isFrameManifestPath(path) or isDirectoryPath(path);
}

fn isRasterImagePath(path: []const u8) bool {
    const extension = std.fs.path.extension(path);
    return std.ascii.eqlIgnoreCase(extension, ".png") or
        std.ascii.eqlIgnoreCase(extension, ".jpg") or
        std.ascii.eqlIgnoreCase(extension, ".jpeg") or
        std.ascii.eqlIgnoreCase(extension, ".bmp") or
        std.ascii.eqlIgnoreCase(extension, ".gif") or
        std.ascii.eqlIgnoreCase(extension, ".ico") or
        std.ascii.eqlIgnoreCase(extension, ".webp");
}

fn isFrameManifestPath(path: []const u8) bool {
    const extension = std.fs.path.extension(path);
    return std.ascii.eqlIgnoreCase(extension, ".frames") or
        std.ascii.eqlIgnoreCase(extension, ".txt") or
        std.ascii.eqlIgnoreCase(extension, ".lst");
}

fn isDirectoryPath(path: []const u8) bool {
    const dir = openDirAtPath(path, .{}) catch return false;
    var opened = dir;
    opened.close(io);
    return true;
}

fn loadPreparedInputFromPath(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (std.ascii.eqlIgnoreCase(std.fs.path.extension(input_path), ".gif")) {
        return try loadPreparedInputFromGif(allocator, input_path, config);
    }
    if (std.ascii.eqlIgnoreCase(std.fs.path.extension(input_path), ".webp")) {
        return try loadPreparedInputFromWebp(allocator, input_path, config);
    }
    if (isRasterImagePath(input_path)) {
        return try preprocess.loadImageInput(allocator, input_path, config);
    }
    if (isDirectoryPath(input_path)) {
        return try loadPreparedInputFromDirectory(allocator, input_path, config);
    }
    if (isFrameManifestPath(input_path)) {
        return try loadPreparedInputFromManifest(allocator, input_path, config);
    }
    return error.UnsupportedImageInput;
}

fn loadPreparedInputFromGif(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) {
        var animation = try imaging.decodeFileGifFramesRgb8(allocator, input_path);
        defer animation.deinit();

        const frame_refs = try allocator.alloc(*const imaging.ImageU8, animation.frames.len);
        defer allocator.free(frame_refs);
        for (animation.frames, frame_refs) |*frame, *slot| {
            slot.* = &frame.image;
        }
        return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
    }
    return try loadPreparedInputFromSingleRaster(allocator, input_path, config);
}

fn loadPreparedInputFromWebp(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) {
        var animation = try imaging.decodeFileWebpFramesRgb8(allocator, input_path);
        defer animation.deinit();

        const frame_refs = try allocator.alloc(*const imaging.ImageU8, animation.frames.len);
        defer allocator.free(frame_refs);
        for (animation.frames, frame_refs) |*frame, *slot| {
            slot.* = &frame.image;
        }
        return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
    }
    return try loadPreparedInputFromSingleRaster(allocator, input_path, config);
}

fn loadPreparedInputFromSingleRaster(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    return try preprocess.loadImageInput(allocator, input_path, config);
}

fn loadPreparedInputFromDirectory(
    allocator: std.mem.Allocator,
    directory_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    var dir = try openDirAtPath(directory_path, .{ .iterate = true });
    defer dir.close(io);

    var entries = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (entries.items) |entry| allocator.free(entry);
        entries.deinit(allocator);
    }

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!isRasterImagePath(entry.name)) continue;
        try entries.append(allocator, try allocator.dupe(u8, entry.name));
    }
    if (entries.items.len == 0) return error.EmptyImageSequence;

    std.sort.block([]u8, entries.items, {}, struct {
        fn lessThan(_: void, lhs: []u8, rhs: []u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    const frame_paths = try allocator.alloc([]const u8, entries.items.len);
    defer allocator.free(frame_paths);
    for (entries.items, frame_paths) |entry, *slot| {
        slot.* = try std.fs.path.join(allocator, &.{ directory_path, entry });
    }
    defer for (frame_paths) |frame_path| allocator.free(frame_path);

    return try loadPreparedInputFromResolvedPaths(allocator, frame_paths, config);
}

fn loadPreparedInputFromManifest(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(bytes);

    const base_dir = std.fs.path.dirname(manifest_path) orelse ".";
    var frame_paths = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (frame_paths.items) |frame_path| allocator.free(frame_path);
        frame_paths.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        const resolved = if (std.fs.path.isAbsolute(line))
            try allocator.dupe(u8, line)
        else
            try std.fs.path.join(allocator, &.{ base_dir, line });
        try frame_paths.append(allocator, resolved);
    }
    if (frame_paths.items.len == 0) return error.EmptyImageSequence;

    return try loadPreparedInputFromResolvedPaths(allocator, frame_paths.items, config);
}

fn loadPreparedInputFromResolvedPaths(
    allocator: std.mem.Allocator,
    frame_paths: []const []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    var images = std.ArrayListUnmanaged(imaging.ImageU8).empty;
    defer images.deinit(allocator);
    errdefer {
        for (images.items) |*image| image.deinit();
    }

    for (frame_paths) |frame_path| {
        try appendResolvedFramesFromPath(allocator, &images, frame_path);
    }
    if (images.items.len == 0) return error.EmptyImageSequence;

    const frame_refs = try allocator.alloc(*const imaging.ImageU8, images.items.len);
    defer allocator.free(frame_refs);
    for (images.items, frame_refs) |*image, *slot| {
        slot.* = image;
    }
    defer for (images.items) |*image| image.deinit();

    return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
}

fn appendResolvedFramesFromPath(
    allocator: std.mem.Allocator,
    images: *std.ArrayListUnmanaged(imaging.ImageU8),
    frame_path: []const u8,
) !void {
    const extension = std.fs.path.extension(frame_path);
    if (std.ascii.eqlIgnoreCase(extension, ".gif") and comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) {
        var animation = try imaging.decodeFileGifFramesRgb8(allocator, frame_path);
        defer animation.deinit();
        for (animation.frames) |*frame| {
            try images.append(allocator, try cloneImageOwned(allocator, &frame.image));
        }
        return;
    }
    if (std.ascii.eqlIgnoreCase(extension, ".webp") and comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) {
        var animation = try imaging.decodeFileWebpFramesRgb8(allocator, frame_path);
        defer animation.deinit();
        for (animation.frames) |*frame| {
            try images.append(allocator, try cloneImageOwned(allocator, &frame.image));
        }
        return;
    }

    try images.append(allocator, try imaging.decodeFileRgb8(allocator, frame_path));
}

fn cloneImageOwned(allocator: std.mem.Allocator, source: *const imaging.ImageU8) !imaging.ImageU8 {
    var cloned = try imaging.ImageU8.init(allocator, source.width, source.height, source.channels);
    errdefer cloned.deinit();
    @memcpy(cloned.data, source.data);
    return cloned;
}

fn openDirAtPath(path: []const u8, flags: std.Io.Dir.OpenOptions) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) return try std.Io.Dir.openDirAbsolute(io, path, flags);
    return try std.Io.Dir.cwd().openDir(io, path, flags);
}

fn hasAnyFile(model_path: []const u8, names: []const []const u8) bool {
    for (names) |name| {
        if (hasFile(model_path, name)) return true;
    }
    return false;
}

fn hasFile(model_path: []const u8, name: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_path, .{}) catch return false;
    defer dir.close(io);

    dir.access(io, name, .{}) catch return false;
    return true;
}

test "native chandra config parser accepts qwen3_5 document vl config" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "architectures": ["Qwen3_5ForConditionalGeneration"],
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "video_token_id": 248057,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 256,
        \\    "vocab_size": 248320,
        \\    "max_position_embeddings": 262144
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 24,
        \\    "hidden_size": 1024,
        \\    "intermediate_size": 4096,
        \\    "num_heads": 16,
        \\    "out_hidden_size": 2560,
        \\    "patch_size": 16,
        \\    "spatial_merge_size": 2,
        \\    "temporal_patch_size": 2,
        \\    "in_channels": 3
        \\  }
        \\}
    );

    const config_path = try tmp.dir.realPathFileAlloc(io, "config.json", std.testing.allocator);
    defer std.testing.allocator.free(config_path);

    var parsed = try loadConfigFromFile(std.testing.allocator, config_path);
    defer parsed.deinit();

    try std.testing.expect(parsed.value.isSupportedChandraShape());
    try std.testing.expectEqual(@as(usize, 248056), parsed.value.image_token_id);
    try std.testing.expectEqual(@as(usize, 24), parsed.value.vision_config.depth);
    try std.testing.expectEqual(@as(usize, 2560), parsed.value.text_config.hidden_size);
}

test "native chandra inspect reports tensor manifest readiness" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 256,
        \\    "vocab_size": 248320,
        \\    "max_position_embeddings": 262144
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 24,
        \\    "hidden_size": 1024,
        \\    "intermediate_size": 4096,
        \\    "num_heads": 16,
        \\    "out_hidden_size": 2560,
        \\    "patch_size": 16,
        \\    "spatial_merge_size": 2,
        \\    "temporal_patch_size": 2,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "merge_size": 2,
        \\  "patch_size": 16,
        \\  "temporal_patch_size": 2,
        \\  "size": {
        \\    "longest_edge": 16777216,
        \\    "shortest_edge": 65536
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "model.safetensors.index.json",
        \\{
        \\  "weight_map": {
        \\    "model.embed_tokens.weight": "model-00001.safetensors",
        \\    "visual.patch_embed.proj.weight": "model-00001.safetensors",
        \\    "visual.merger.mlp.0.weight": "model-00001.safetensors",
        \\    "lm_head.weight": "model-00001.safetensors"
        \\  }
        \\}
    );

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);

    const readiness = inspect(root_path);
    try std.testing.expect(readiness.has_config);
    try std.testing.expect(readiness.has_supported_config);
    try std.testing.expect(readiness.has_weights);
    try std.testing.expect(readiness.has_visual_encoder);
    try std.testing.expect(readiness.has_patch_embedding_weight);
    try std.testing.expect(readiness.has_multimodal_projector);
    try std.testing.expect(readiness.has_image_processor_config);
    try std.testing.expect(readiness.has_document_preprocessor);
    try std.testing.expectEqual(@as(usize, 1), readiness.text_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.vision_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.projector_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.output_tensor_count);
}

test "native chandra execute preprocesses image and runs patch embedding stage" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 32,
        \\    "max_position_embeddings": 1024
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 2,
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_heads": 1,
        \\    "out_hidden_size": 2,
        \\    "patch_size": 2,
        \\    "spatial_merge_size": 1,
        \\    "temporal_patch_size": 1,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "do_normalize": false,
        \\  "do_rescale": false,
        \\  "do_resize": false,
        \\  "merge_size": 1,
        \\  "patch_size": 2,
        \\  "temporal_patch_size": 1,
        \\  "size": {
        \\    "longest_edge": 1024,
        \\    "shortest_edge": 1
        \\  }
        \\}
    );
    try writeSyntheticPatchEmbeddingSafetensors(tmp.dir, "model.safetensors");

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"visual_merger\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_embedding_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_embedding_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_position_embedding_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_position_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"vision_block_depth\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_block_attention_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_attention_blocks_executed\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_attention_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_block_mlp_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_mlp_blocks_executed\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_merger_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_token_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_token_count\":2") != null);
}

test "chandra input loader prepares frame sequence from directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeSolidPng(tmp.dir, "frame_02.png", 2, 2, 255);
    try writeSolidPng(tmp.dir, "frame_01.png", 2, 2, 0);

    const frames_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(frames_path);

    var prepared = try loadPreparedInputFromDirectory(std.testing.allocator, frames_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 2), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.temporal_patch_count);
    try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[0]);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
}

test "chandra input loader prepares frame sequence from manifest" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeSolidPng(tmp.dir, "a.png", 2, 2, 64);
    try writeSolidPng(tmp.dir, "b.png", 2, 2, 192);
    try writeTmpFile(tmp.dir, "frames.frames",
        \\# ordered frame list
        \\b.png
        \\a.png
    );

    const manifest_path = try tmp.dir.realPathFileAlloc(io, "frames.frames", std.testing.allocator);
    defer std.testing.allocator.free(manifest_path);

    var prepared = try loadPreparedInputFromManifest(std.testing.allocator, manifest_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 2), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 192.0), prepared.tensor.data[0]);
    try std.testing.expectEqual(@as(f32, 64.0), prepared.tensor.data[prepared.tensor.stride_n]);
}

test "chandra input loader handles gif via pixio codec" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedGif(tmp.dir, "animated.gif");
    const gif_path = try tmp.dir.realPathFileAlloc(io, "animated.gif", std.testing.allocator);
    defer std.testing.allocator.free(gif_path);

    var prepared = try loadPreparedInputFromPath(std.testing.allocator, gif_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) 2 else 1;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 2) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    } else {
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    }
}

test "chandra input loader handles webp via pixio codec" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedWebp(tmp.dir, "animated.webp");
    const webp_path = try tmp.dir.realPathFileAlloc(io, "animated.webp", std.testing.allocator);
    defer std.testing.allocator.free(webp_path);

    var prepared = try loadPreparedInputFromPath(std.testing.allocator, webp_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) 2 else 1;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 2) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    } else {
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    }
}

test "chandra directory loader expands animated webp entries into frame sequence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedWebp(tmp.dir, "000.webp");
    try writeSolidPng(tmp.dir, "001.png", 1, 1, 32);
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir_path);

    var prepared = try loadPreparedInputFromPath(std.testing.allocator, dir_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 4,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) 3 else 2;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 3) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 32.0), prepared.tensor.data[prepared.tensor.stride_n * 2]);
    } else {
        try std.testing.expectEqual(@as(f32, 32.0), prepared.tensor.data[prepared.tensor.stride_n]);
    }
}

test "chandra manifest loader expands animated gif entries into frame sequence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedGif(tmp.dir, "clip.gif");
    try writeSolidPng(tmp.dir, "tail.png", 1, 1, 48);
    try writeTmpFile(tmp.dir, "frames.lst",
        \\clip.gif
        \\tail.png
        \\
    );
    const manifest_path = try tmp.dir.realPathFileAlloc(io, "frames.lst", std.testing.allocator);
    defer std.testing.allocator.free(manifest_path);

    var prepared = try loadPreparedInputFromPath(std.testing.allocator, manifest_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 4,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) 3 else 2;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 3) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 48.0), prepared.tensor.data[prepared.tensor.stride_n * 2]);
    } else {
        try std.testing.expectEqual(@as(f32, 48.0), prepared.tensor.data[prepared.tensor.stride_n]);
    }
}

test "multimodal mrope plan continues text after visual axis max" {
    const position_plan = MultimodalPositionPlan.init(.mrope, 6, 1, 3, 2, 2);

    try std.testing.expectEqual(@as(usize, 0), position_plan.vision_start_position);
    try std.testing.expectEqual(@as(usize, 1), position_plan.visual_start_position);
    try std.testing.expectEqual(@as(usize, 4), position_plan.vision_end_position);
    try std.testing.expectEqual(@as(usize, 5), position_plan.prompt_start_position);
    try std.testing.expectEqual(@as(usize, 7), position_plan.generation_start_position);
    try std.testing.expectEqual(@as(usize, 10), position_plan.total_prefill_tokens);

    const first_visual = visualTokenPosition(.mrope, position_plan.visual_start_position, 0, 1, 2, 3);
    try std.testing.expectEqual(decoder_types.RopePositionMode.mrope, first_visual.mode);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, &first_visual.axes);

    const tail_visual = visualTokenPosition(.mrope, position_plan.visual_start_position, 5, 1, 2, 3);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 3 }, &tail_visual.axes);

    const prompt_position = textTokenPosition(.mrope, position_plan.prompt_start_position);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5, 5 }, &prompt_position.axes);
}

test "visual token position maps thw axes for future multi-frame layout" {
    const position = visualTokenPosition(.mrope, 4, 7, 2, 2, 2);
    try std.testing.expectEqual(decoder_types.RopePositionMode.mrope, position.mode);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5, 5 }, &position.axes);
}

test "native chandra execute exposes mrope prefill metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 3,
        \\  "vision_start_token_id": 1,
        \\  "vision_end_token_id": 2,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 8,
        \\    "max_position_embeddings": 1024,
        \\    "rope_parameters": {
        \\      "full_attention": {
        \\        "rope_theta": 250000.0
        \\      },
        \\      "mrope_section": [1, 0, 0]
        \\    }
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 2,
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_heads": 1,
        \\    "out_hidden_size": 2,
        \\    "patch_size": 2,
        \\    "spatial_merge_size": 1,
        \\    "temporal_patch_size": 1,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "do_normalize": false,
        \\  "do_rescale": false,
        \\  "do_resize": false,
        \\  "merge_size": 1,
        \\  "patch_size": 2,
        \\  "temporal_patch_size": 1,
        \\  "size": {
        \\    "longest_edge": 1024,
        \\    "shortest_edge": 1
        \\  }
        \\}
    );
    try writeSyntheticRuntimeReadySafetensors(tmp.dir, "model.safetensors");

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
        .max_output_tokens = 0,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"text_prefill\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"text_prefill_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoder_rope_position_mode\":\"mrope\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoder_mrope_sections\":[1,0,0,0]") != null);
}

test "native chandra execute decodes content with synthetic tokenizer" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 3,
        \\  "vision_start_token_id": 1,
        \\  "vision_end_token_id": 2,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 8,
        \\    "max_position_embeddings": 1024,
        \\    "rope_parameters": {
        \\      "full_attention": {
        \\        "rope_theta": 250000.0
        \\      },
        \\      "mrope_section": [1, 0, 0]
        \\    }
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 2,
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_heads": 1,
        \\    "out_hidden_size": 2,
        \\    "patch_size": 2,
        \\    "spatial_merge_size": 1,
        \\    "temporal_patch_size": 1,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "do_normalize": false,
        \\  "do_rescale": false,
        \\  "do_resize": false,
        \\  "merge_size": 1,
        \\  "patch_size": 2,
        \\  "temporal_patch_size": 1,
        \\  "size": {
        \\    "longest_edge": 1024,
        \\    "shortest_edge": 1
        \\  }
        \\}
    );
    try writeSyntheticRuntimeReadySafetensors(tmp.dir, "model.safetensors");
    try writeSyntheticTokenizerFiles(tmp.dir);

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
        .max_output_tokens = 1,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"text_decode\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"ocr_native_text_decoded_partial\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"text_decode_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoded_token_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"content\":\"OCR\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"markdown\":\"OCR\"") != null);
}

fn writeTmpFile(dir: std.Io.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);

    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;
    try writer.writeAll(contents);
    try writer.flush();
}

fn writeSolidPng(dir: std.Io.Dir, relative_path: []const u8, width: usize, height: usize, value: u8) !void {
    var image = try imaging.ImageU8.init(std.testing.allocator, width, height, 3);
    defer image.deinit();
    image.fill(value);

    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = encoded });
}

fn writeAnimatedGif(dir: std.Io.Dir, relative_path: []const u8) !void {
    const gif_bytes = [_]u8{
        'G',  'I',  'F',  '8',  '9',  'a',
        0x01, 0x00, 0x01, 0x00, 0x80, 0x00,
        0x00, 0xff, 0x00, 0x00, 0x00, 0xff,
        0x00, 0x21, 0xf9, 0x04, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x00, 0x02, 0x02, 0x44, 0x01, 0x00,
        0x21, 0xf9, 0x04, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x2c, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0x02, 0x02, 0x4c, 0x01, 0x00, 0x3b,
    };
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = &gif_bytes });
}

fn writeAnimatedWebp(dir: std.Io.Dir, relative_path: []const u8) !void {
    const webp_bytes = [_]u8{
        0x52, 0x49, 0x46, 0x46, 0x84, 0x00, 0x00, 0x00,
        0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58,
        0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x4e,
        0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46,
        0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c,
        0x0f, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
        0x00, 0x07, 0x10, 0xfd, 0x8f, 0xfe, 0x07, 0x22,
        0xa2, 0xff, 0x01, 0x00, 0x41, 0x4e, 0x4d, 0x46,
        0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc8, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c,
        0x0f, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
        0x00, 0x07, 0xd0, 0xff, 0x88, 0xfe, 0x07, 0x22,
        0xa2, 0xff, 0x01, 0x00,
    };
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = &webp_bytes });
}

fn writeSyntheticPatchEmbeddingSafetensors(dir: std.Io.Dir, relative_path: []const u8) !void {
    const header =
        \\{"visual.patch_embed.proj.weight":{"dtype":"F32","shape":[2,3,1,2,2],"data_offsets":[0,96]},"visual.patch_embed.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[96,104]},"visual.pos_embed":{"dtype":"F32","shape":[4,2],"data_offsets":[104,136]},"visual.blocks.0.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[136,144]},"visual.blocks.0.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[144,152]},"visual.blocks.0.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[152,200]},"visual.blocks.0.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[200,224]},"visual.blocks.0.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[224,240]},"visual.blocks.0.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[240,248]},"visual.blocks.0.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[248,256]},"visual.blocks.0.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[256,264]},"visual.blocks.0.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[264,288]},"visual.blocks.0.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[288,300]},"visual.blocks.0.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[300,324]},"visual.blocks.0.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[324,332]},"visual.blocks.1.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[332,340]},"visual.blocks.1.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[340,348]},"visual.blocks.1.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[348,396]},"visual.blocks.1.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[396,420]},"visual.blocks.1.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[420,436]},"visual.blocks.1.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[436,444]},"visual.blocks.1.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[444,452]},"visual.blocks.1.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[452,460]},"visual.blocks.1.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[460,484]},"visual.blocks.1.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[484,496]},"visual.blocks.1.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[496,520]},"visual.blocks.1.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[520,528]},"visual.merger.mlp.0.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[528,552]},"visual.merger.mlp.0.bias":{"dtype":"F32","shape":[3],"data_offsets":[552,564]},"visual.merger.mlp.2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[564,588]},"visual.merger.mlp.2.bias":{"dtype":"F32","shape":[2],"data_offsets":[588,596]},"model.embed_tokens.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[596,604]},"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[604,612]}}
    ;

    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);
    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try writer.writeAll(&length_prefix);
    try writer.writeAll(header);

    var payload: [612]u8 = undefined;
    @memset(&payload, 0);

    const patch_weights = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    writeF32Slice(&payload, 0, &patch_weights);
    writeF32Scalar(&payload, 96, 0.0);
    writeF32Scalar(&payload, 100, 1.0);

    const visual_pos = [_]f32{
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    };
    writeF32Slice(&payload, 104, &visual_pos);

    const block_qkv = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    const block_proj = [_]f32{
        1, 0,
        0, 1,
    };
    const block_norm_weight = [_]f32{ 1, 1 };
    const block_norm_bias = [_]f32{ 0, 0 };
    const block_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const block_fc1_bias = [_]f32{ 0, 0, 0.5 };
    const block_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const block_proj_bias = [_]f32{ 0, 0 };
    const block_qkv_bias = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const block_fc2_bias = [_]f32{ 0, 0 };

    writeSyntheticVisionBlock(&payload, 136, block_norm_weight[0..], block_norm_bias[0..], block_qkv[0..], block_qkv_bias[0..], block_proj[0..], block_proj_bias[0..], block_norm_weight[0..], block_norm_bias[0..], block_fc1[0..], block_fc1_bias[0..], block_fc2[0..], block_fc2_bias[0..]);
    writeSyntheticVisionBlock(&payload, 332, block_norm_weight[0..], block_norm_bias[0..], block_qkv[0..], block_qkv_bias[0..], block_proj[0..], block_proj_bias[0..], block_norm_weight[0..], block_norm_bias[0..], block_fc1[0..], block_fc1_bias[0..], block_fc2[0..], block_fc2_bias[0..]);

    const merger_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    writeF32Slice(&payload, 528, &merger_fc1);
    writeF32Scalar(&payload, 552, 0.0);
    writeF32Scalar(&payload, 556, 0.0);
    writeF32Scalar(&payload, 560, 0.5);

    const merger_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    writeF32Slice(&payload, 564, &merger_fc2);
    writeF32Scalar(&payload, 588, 0.25);
    writeF32Scalar(&payload, 592, -0.25);

    try writer.writeAll(&payload);
    try writer.flush();
}

const SyntheticTensorSpec = struct {
    name: []const u8,
    shape: []const usize,
    values: []const f32,
};

fn writeSyntheticRuntimeReadySafetensors(dir: std.Io.Dir, relative_path: []const u8) !void {
    const patch_weight_shape = [_]usize{ 2, 3, 1, 2, 2 };
    const patch_bias_shape = [_]usize{2};
    const pos_shape = [_]usize{ 4, 2 };
    const qkv_shape = [_]usize{ 6, 2 };
    const linear_2x2_shape = [_]usize{ 2, 2 };
    const mlp_fc1_shape = [_]usize{ 3, 2 };
    const mlp_fc2_shape = [_]usize{ 2, 3 };
    const embed_shape = [_]usize{ 8, 2 };
    const norm_shape = [_]usize{2};
    const gate_shape = [_]usize{ 8, 2 };
    const down_shape = [_]usize{ 2, 8 };

    const patch_weights = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    const patch_bias = [_]f32{ 0, 1 };
    const visual_pos = [_]f32{
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    };
    const block_norm_weight = [_]f32{ 1, 1 };
    const block_norm_bias = [_]f32{ 0, 0 };
    const block_qkv = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    const block_qkv_bias = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const block_proj = [_]f32{
        1, 0,
        0, 1,
    };
    const block_proj_bias = [_]f32{ 0, 0 };
    const block_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const block_fc1_bias = [_]f32{ 0, 0, 0.5 };
    const block_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const block_fc2_bias = [_]f32{ 0, 0 };
    const merger_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const merger_fc1_bias = [_]f32{ 0, 0, 0.5 };
    const merger_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const merger_fc2_bias = [_]f32{ 0.25, -0.25 };
    const embed_tokens = [_]f32{
        0,   0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0.5, 0.5,
        0.6, 0.6,
        0.7, 0.7,
    };
    const final_norm = [_]f32{ 1, 1 };
    const q_norm = [_]f32{ 1, 1 };
    const zero_2x2 = [_]f32{0} ** 4;
    const zero_gate = [_]f32{0} ** 16;
    const zero_down = [_]f32{0} ** 16;
    const lm_head = [_]f32{
        0,  0,
        0,  0,
        0,  0,
        0,  0,
        10, 10,
        0,  0,
        0,  0,
        0,  0,
    };

    const specs = [_]SyntheticTensorSpec{
        .{ .name = "visual.patch_embed.proj.weight", .shape = &patch_weight_shape, .values = &patch_weights },
        .{ .name = "visual.patch_embed.proj.bias", .shape = &patch_bias_shape, .values = &patch_bias },
        .{ .name = "visual.pos_embed", .shape = &pos_shape, .values = &visual_pos },
        .{ .name = "visual.blocks.0.norm1.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.0.norm1.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.0.attn.qkv.weight", .shape = &qkv_shape, .values = &block_qkv },
        .{ .name = "visual.blocks.0.attn.qkv.bias", .shape = &[_]usize{6}, .values = &block_qkv_bias },
        .{ .name = "visual.blocks.0.attn.proj.weight", .shape = &linear_2x2_shape, .values = &block_proj },
        .{ .name = "visual.blocks.0.attn.proj.bias", .shape = &norm_shape, .values = &block_proj_bias },
        .{ .name = "visual.blocks.0.norm2.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.0.norm2.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.0.mlp.linear_fc1.weight", .shape = &mlp_fc1_shape, .values = &block_fc1 },
        .{ .name = "visual.blocks.0.mlp.linear_fc1.bias", .shape = &[_]usize{3}, .values = &block_fc1_bias },
        .{ .name = "visual.blocks.0.mlp.linear_fc2.weight", .shape = &mlp_fc2_shape, .values = &block_fc2 },
        .{ .name = "visual.blocks.0.mlp.linear_fc2.bias", .shape = &norm_shape, .values = &block_fc2_bias },
        .{ .name = "visual.blocks.1.norm1.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.1.norm1.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.1.attn.qkv.weight", .shape = &qkv_shape, .values = &block_qkv },
        .{ .name = "visual.blocks.1.attn.qkv.bias", .shape = &[_]usize{6}, .values = &block_qkv_bias },
        .{ .name = "visual.blocks.1.attn.proj.weight", .shape = &linear_2x2_shape, .values = &block_proj },
        .{ .name = "visual.blocks.1.attn.proj.bias", .shape = &norm_shape, .values = &block_proj_bias },
        .{ .name = "visual.blocks.1.norm2.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.1.norm2.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.1.mlp.linear_fc1.weight", .shape = &mlp_fc1_shape, .values = &block_fc1 },
        .{ .name = "visual.blocks.1.mlp.linear_fc1.bias", .shape = &[_]usize{3}, .values = &block_fc1_bias },
        .{ .name = "visual.blocks.1.mlp.linear_fc2.weight", .shape = &mlp_fc2_shape, .values = &block_fc2 },
        .{ .name = "visual.blocks.1.mlp.linear_fc2.bias", .shape = &norm_shape, .values = &block_fc2_bias },
        .{ .name = "visual.merger.mlp.0.weight", .shape = &mlp_fc1_shape, .values = &merger_fc1 },
        .{ .name = "visual.merger.mlp.0.bias", .shape = &[_]usize{3}, .values = &merger_fc1_bias },
        .{ .name = "visual.merger.mlp.2.weight", .shape = &mlp_fc2_shape, .values = &merger_fc2 },
        .{ .name = "visual.merger.mlp.2.bias", .shape = &norm_shape, .values = &merger_fc2_bias },
        .{ .name = "model.embed_tokens.weight", .shape = &embed_shape, .values = &embed_tokens },
        .{ .name = "model.norm.weight", .shape = &norm_shape, .values = &final_norm },
        .{ .name = "lm_head.weight", .shape = &embed_shape, .values = &lm_head },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "model.layers.0.self_attn.q_norm.weight", .shape = &norm_shape, .values = &q_norm },
        .{ .name = "model.layers.0.self_attn.k_norm.weight", .shape = &norm_shape, .values = &q_norm },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &gate_shape, .values = &zero_gate },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &gate_shape, .values = &zero_gate },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &down_shape, .values = &zero_down },
    };

    try writeSyntheticF32Safetensors(std.testing.allocator, dir, relative_path, &specs);
}

fn writeSyntheticTokenizerFiles(dir: std.Io.Dir) !void {
    try writeTmpFile(dir, "vocab.json", "{}");
    try writeTmpFile(dir, "merges.txt", "# synthetic\n");
    try writeTmpFile(dir, "tokenizer_config.json",
        \\{
        \\  "added_tokens_decoder": {
        \\    "4": { "content": "OCR" },
        \\    "7": {
        \\      "content": "<|im_start|>user\nRead the document image and transcribe it as markdown.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        \\    }
        \\  }
        \\}
    );
}

fn writeSyntheticF32Safetensors(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    relative_path: []const u8,
    specs: []const SyntheticTensorSpec,
) !void {
    var header = std.ArrayList(u8).init(allocator);
    defer header.deinit();

    var offsets = try allocator.alloc(u64, specs.len + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (specs, 0..) |spec, index| {
        const tensor_elements = try tensorElementCount(spec.shape);
        if (tensor_elements != spec.values.len) return error.ShapeMismatch;
        offsets[index + 1] = offsets[index] + spec.values.len * @sizeOf(f32);
    }

    try header.append('{');
    for (specs, 0..) |spec, index| {
        if (index != 0) try header.append(',');
        try header.writer().print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{spec.name});
        for (spec.shape, 0..) |dim, dim_index| {
            if (dim_index != 0) try header.append(',');
            try header.writer().print("{d}", .{dim});
        }
        try header.writer().print("],\"data_offsets\":[{d},{d}]}}", .{
            offsets[index],
            offsets[index + 1],
        });
    }
    try header.append('}');

    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);
    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.items.len, .little);
    try writer.writeAll(&length_prefix);
    try writer.writeAll(header.items);

    for (specs) |spec| {
        for (spec.values) |value| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
            try writer.writeAll(&bytes);
        }
    }
    try writer.flush();
}

fn tensorElementCount(shape: []const usize) !usize {
    var total: usize = 1;
    for (shape) |dim| {
        total = try std.math.mul(usize, total, dim);
    }
    return total;
}

fn writeSyntheticVisionBlock(
    payload: []u8,
    offset: usize,
    norm1_weight: []const f32,
    norm1_bias: []const f32,
    qkv_weight: []const f32,
    qkv_bias: []const f32,
    proj_weight: []const f32,
    proj_bias: []const f32,
    norm2_weight: []const f32,
    norm2_bias: []const f32,
    fc1_weight: []const f32,
    fc1_bias: []const f32,
    fc2_weight: []const f32,
    fc2_bias: []const f32,
) void {
    writeF32Slice(payload, offset, norm1_weight);
    writeF32Slice(payload, offset + 8, norm1_bias);
    writeF32Slice(payload, offset + 16, qkv_weight);
    writeF32Slice(payload, offset + 64, qkv_bias);
    writeF32Slice(payload, offset + 88, proj_weight);
    writeF32Slice(payload, offset + 104, proj_bias);
    writeF32Slice(payload, offset + 112, norm2_weight);
    writeF32Slice(payload, offset + 120, norm2_bias);
    writeF32Slice(payload, offset + 128, fc1_weight);
    writeF32Slice(payload, offset + 152, fc1_bias);
    writeF32Slice(payload, offset + 164, fc2_weight);
    writeF32Slice(payload, offset + 188, fc2_bias);
}

fn writeF32Slice(payload: []u8, offset: usize, values: []const f32) void {
    for (values, 0..) |value, index| {
        writeF32Scalar(payload, offset + index * 4, value);
    }
}

fn writeF32Scalar(payload: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, payload[offset .. offset + 4][0..4], @bitCast(value), .little);
}
