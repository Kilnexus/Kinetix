const std = @import("std");
const preprocess = @import("../../../chandra_preprocess.zig");
const store = @import("../../../chandra_store.zig");
const vision = @import("../../../chandra_vision.zig");
const core = @import("../model/core.zig");
const input = @import("../input/loader.zig");
const decoder_runtime = @import("../../../../text/decoder_runtime.zig");
const decoder_family = @import("../../../../text/decoder_family.zig");
const decoder_types = @import("../../../../text/decoder_types.zig");
const text_backend_scheme = @import("../../../../text/backend_scheme.zig");
const kv_cache = @import("../../../../text/kv_cache.zig");
const streaming = @import("../../../../text/streaming.zig");

pub const MaterializedOutput = union(enum) {
    text: []u8,
    json: []u8,

    pub fn deinit(self: *MaterializedOutput, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .text => |value| allocator.free(value),
            .json => |value| allocator.free(value),
        }
        self.* = undefined;
    }
};

pub const NativeExecutionResult = struct {
    readiness: core.Readiness,
    summary: ?core.PreprocessSummary = null,

    pub fn deinit(self: *NativeExecutionResult, allocator: std.mem.Allocator) void {
        if (self.summary) |*value| value.deinit(allocator);
        self.* = undefined;
    }

    pub fn materializeOutput(
        self: *const NativeExecutionResult,
        allocator: std.mem.Allocator,
        operation: []const u8,
    ) !?MaterializedOutput {
        const summary = self.summary orelse return null;
        const generated = summary.generated_output orelse return null;

        if (std.mem.eql(u8, operation, "render-json")) {
            return .{ .json = try allocator.dupe(u8, generated) };
        }
        return .{ .text = try allocator.dupe(u8, generated) };
    }

    pub fn toJsonAlloc(
        self: *const NativeExecutionResult,
        allocator: std.mem.Allocator,
        context: core.Context,
    ) ![]u8 {
        return try buildOutputJson(allocator, context, self.readiness, self.summary);
    }
};

const ExecutionResources = struct {
    allocator: std.mem.Allocator,
    image_processor: ?*const preprocess.ParsedImageProcessorConfig = null,
    parsed_config: ?*const core.ParsedConfig = null,
    tensor_store: ?*const store.ChandraStore = null,
    owned_image_processor: ?preprocess.ParsedImageProcessorConfig = null,
    owned_parsed_config: ?core.ParsedConfig = null,
    owned_tensor_store: ?store.ChandraStore = null,

    fn init(
        allocator: std.mem.Allocator,
        context: core.Context,
        loaded_model: core.LoadedModel,
    ) !ExecutionResources {
        var resources: ExecutionResources = .{
            .allocator = allocator,
            .image_processor = loaded_model.image_processor,
            .parsed_config = loaded_model.parsed_config,
            .tensor_store = loaded_model.tensor_store,
        };
        errdefer resources.deinit();

        if (resources.image_processor == null) {
            resources.owned_image_processor = try preprocess.loadImageProcessorConfig(allocator, context.model_path);
            if (resources.owned_image_processor) |*value| resources.image_processor = value;
        }

        if (resources.parsed_config == null) {
            const config_path = std.fs.path.join(allocator, &.{ context.model_path, "config.json" }) catch null;
            defer if (config_path) |path| allocator.free(path);
            if (config_path) |path| {
                resources.owned_parsed_config = core.loadConfigFromFile(allocator, path) catch null;
                if (resources.owned_parsed_config) |*value| resources.parsed_config = value;
            }
        }

        if (resources.tensor_store == null and loaded_model.readiness.has_patch_embedding_weight) {
            resources.owned_tensor_store = store.ChandraStore.open(allocator, context.model_path) catch null;
            if (resources.owned_tensor_store) |*value| resources.tensor_store = value;
        }

        return resources;
    }

    fn deinit(self: *ExecutionResources) void {
        if (self.owned_tensor_store) |*value| value.deinit();
        if (self.owned_parsed_config) |*value| value.deinit();
        if (self.owned_image_processor) |*value| value.deinit();
        self.* = undefined;
    }
};

pub fn execute(allocator: std.mem.Allocator, context: core.Context) ![]u8 {
    var result = try executeDetailedWithLoadedModel(allocator, context, .{
        .readiness = core.inspect(context.model_path),
    });
    defer result.deinit(allocator);
    return try result.toJsonAlloc(allocator, context);
}

pub fn executeWithLoadedModel(
    allocator: std.mem.Allocator,
    context: core.Context,
    loaded_model: core.LoadedModel,
) ![]u8 {
    var result = try executeDetailedWithLoadedModel(allocator, context, loaded_model);
    defer result.deinit(allocator);
    return try result.toJsonAlloc(allocator, context);
}

pub fn executeDetailed(
    allocator: std.mem.Allocator,
    context: core.Context,
) !NativeExecutionResult {
    return try executeDetailedWithLoadedModel(allocator, context, .{
        .readiness = core.inspect(context.model_path),
    });
}

pub fn executeDetailedWithLoadedModel(
    allocator: std.mem.Allocator,
    context: core.Context,
    loaded_model: core.LoadedModel,
) !NativeExecutionResult {
    if (context.execution != .sync) return error.UnsupportedExecutionMode;

    var summary: ?core.PreprocessSummary = null;
    errdefer if (summary) |*value| value.deinit(allocator);

    if (loaded_model.readiness.has_image_processor_config and input.isSupportedInputPath(context.input_path)) {
        var resources = try ExecutionResources.init(allocator, context, loaded_model);
        defer resources.deinit();

        const image_processor = resources.image_processor orelse return error.MissingImageProcessorConfig;
        var prepared = try input.loadPreparedInputFromPath(allocator, context.input_path, image_processor.value);
        defer prepared.deinit();

        const vision_block_depth = if (resources.parsed_config) |config| config.value.vision_config.depth else 0;
        summary = initSummary(&prepared, vision_block_depth);
        try runVisionPipeline(
            allocator,
            context,
            &summary.?,
            image_processor.value,
            resources.parsed_config,
            resources.tensor_store,
            &prepared,
        );
    }

    return .{
        .readiness = loaded_model.readiness,
        .summary = summary,
    };
}

fn initSummary(prepared: *const preprocess.PreparedImageInput, vision_block_depth: usize) core.PreprocessSummary {
    return .{
        .image_width = prepared.grid.input_width,
        .image_height = prepared.grid.input_height,
        .resized_width = prepared.grid.resized_width,
        .resized_height = prepared.grid.resized_height,
        .patch_token_count = prepared.grid.patch_token_count,
        .visual_token_count = prepared.grid.token_count,
        .vision_block_depth = vision_block_depth,
    };
}

fn runVisionPipeline(
    allocator: std.mem.Allocator,
    context: core.Context,
    summary: *core.PreprocessSummary,
    image_processor_config: preprocess.ImageProcessorConfig,
    parsed_config: ?*const core.ParsedConfig,
    tensor_store: ?*const store.ChandraStore,
    prepared: *const preprocess.PreparedImageInput,
) !void {
    const opened = tensor_store orelse return;

    var patch_weights = opened.loadPatchEmbeddingWeights(allocator) catch return;
    defer patch_weights.deinit();

    var embeddings = vision.patchEmbedImage(allocator, prepared, patch_weights.weights) catch return;
    defer embeddings.deinit();
    summary.patch_embedding_dim = embeddings.embedding_dim;
    summary.patch_embedding_executed = true;

    var current = embeddings;
    var current_active = true;
    defer if (current_active) current.deinit();

    var pos_weights = opened.loadVisualPositionEmbeddings(allocator, current.embedding_dim) catch null;
    if (pos_weights) |*pos| {
        defer pos.deinit();
        const positioned = vision.applyPositionEmbeddings(allocator, current, pos.weights) catch null;
        if (positioned) |next| {
            current.deinit();
            current = next;
            summary.visual_position_dim = current.embedding_dim;
            summary.visual_position_embedding_executed = true;
        }
    }

    const vision_num_heads = if (parsed_config) |config|
        @max(@as(usize, 1), config.value.vision_config.num_heads)
    else
        1;

    for (0..summary.vision_block_depth) |block_index| {
        var attn = opened.loadVisionBlockAttentionWeights(allocator, block_index, vision_num_heads) catch break;
        defer attn.deinit();
        const attn_next = vision.applyVisionBlockAttention(allocator, current, attn.weights, 1e-5) catch break;
        current.deinit();
        current = attn_next;
        summary.visual_attention_dim = current.embedding_dim;
        summary.visual_block_attention_executed = true;
        summary.visual_attention_blocks_executed = block_index + 1;

        var mlp = opened.loadVisionBlockMlpWeights(allocator, block_index) catch break;
        defer mlp.deinit();
        const mlp_next = vision.applyVisionBlockMlp(allocator, current, mlp.weights, 1e-5) catch break;
        current.deinit();
        current = mlp_next;
        summary.visual_block_dim = current.embedding_dim;
        summary.visual_block_mlp_executed = true;
        summary.visual_mlp_blocks_executed = block_index + 1;
    }

    var grouped = vision.mergeSpatialPatches(allocator, current, image_processor_config.merge_size) catch return;
    defer grouped.deinit();

    var merger_weights = opened.loadVisualMergerWeights(allocator) catch return;
    defer merger_weights.deinit();

    var tokens = vision.applyVisualMerger(allocator, grouped, merger_weights.weights) catch return;
    defer tokens.deinit();

    summary.visual_token_dim = tokens.embedding_dim;
    summary.visual_merger_executed = true;

    if (parsed_config) |config| {
        try runTextPipeline(allocator, context, summary, config, &tokens);
    }

    current_active = false;
}

fn runTextPipeline(
    allocator: std.mem.Allocator,
    context: core.Context,
    summary: *core.PreprocessSummary,
    parsed_config: *const core.ParsedConfig,
    tokens: *const vision.VisualTokens,
) !void {
    var text_runtime = decoder_runtime.initRuntime(
        allocator,
        context.model_path,
        text_backend_scheme.Scheme.auto,
        1,
    ) catch return;
    defer text_runtime.deinit();

    summary.decoder_rope_position_mode = text_runtime.cfg.rope_position_mode.name();
    summary.decoder_mrope_sections = text_runtime.cfg.mrope_sections;

    var tokenizer = decoder_family.loadTokenizerFromModelDir(
        allocator,
        text_runtime.cfg.architecture,
        context.model_path,
    ) catch null;
    defer if (tokenizer) |*loaded| loaded.deinit();

    const prompt_token_ids = try encodePromptTokenIds(allocator, context.operation, &text_runtime, if (tokenizer) |*loaded| loaded else null);
    defer if (prompt_token_ids) |ids| allocator.free(ids);
    if (prompt_token_ids) |ids| summary.text_prompt_token_count = ids.len;

    const position_plan = core.MultimodalPositionPlan.init(
        text_runtime.cfg.rope_position_mode,
        tokens.token_count,
        tokens.grid_time,
        tokens.grid_width,
        tokens.grid_height,
        summary.text_prompt_token_count,
    );
    const resolved_kv_cache_scheme = kv_cache.resolveScheme(.auto, text_runtime.backendName());

    var cache = kv_cache.ModelCache.initWithLayout(
        allocator,
        text_runtime.cfg.num_hidden_layers,
        position_plan.total_prefill_tokens + (context.max_output_tokens orelse 64),
        text_runtime.cfg.num_key_value_heads,
        text_runtime.cfg.head_dim,
        resolved_kv_cache_scheme,
        kv_cache.default_q8_layout,
    ) catch return;
    defer cache.deinit();

    var workspace = text_runtime.initWorkspace(position_plan.total_prefill_tokens + (context.max_output_tokens orelse 64)) catch return;
    defer workspace.deinit();

    const current_logits = text_runtime.forwardTokenIdWithPosition(
        &workspace,
        &cache,
        parsed_config.value.vision_start_token_id,
        core.textTokenPosition(text_runtime.cfg.rope_position_mode, position_plan.vision_start_position),
    ) catch return;
    var latest_logits = current_logits;

    const visual_positions = core.allocMultimodalVisualPositions(
        allocator,
        text_runtime.cfg.rope_position_mode,
        position_plan.visual_start_position,
        tokens.token_count,
        tokens.grid_time,
        tokens.grid_height,
        tokens.grid_width,
    ) catch null;
    defer if (visual_positions) |positions| allocator.free(positions);
    if (visual_positions) |positions| {
        latest_logits = text_runtime.prefillEmbeddingsWithPositions(
            &workspace,
            &cache,
            tokens.data,
            tokens.token_count,
            positions,
        ) catch latest_logits;
    }

    latest_logits = text_runtime.forwardTokenIdWithPosition(
        &workspace,
        &cache,
        parsed_config.value.vision_end_token_id,
        core.textTokenPosition(text_runtime.cfg.rope_position_mode, position_plan.vision_end_position),
    ) catch latest_logits;

    if (prompt_token_ids) |ids| {
        const prompt_positions = core.allocMultimodalTextPositions(
            allocator,
            text_runtime.cfg.rope_position_mode,
            position_plan.prompt_start_position,
            ids.len,
        ) catch null;
        defer if (prompt_positions) |positions| allocator.free(positions);
        if (prompt_positions) |positions| {
            latest_logits = text_runtime.prefillTokenIdsWithPositions(
                &workspace,
                &cache,
                ids,
                positions,
            ) catch latest_logits;
        }
    }

    summary.text_prefill_token_count = position_plan.total_prefill_tokens;
    summary.text_prefill_executed = true;
    summary.decoder_logits_dim = latest_logits.len;
    summary.decoder_next_token_id = decoder_family.argMaxLogit(latest_logits) catch null;

    if (tokenizer) |*loaded_tokenizer| {
        var generated_ids = std.ArrayListUnmanaged(u32).empty;
        defer generated_ids.deinit(allocator);

        var step: usize = 0;
        const max_output_tokens = context.max_output_tokens orelse 64;
        while (step < max_output_tokens) : (step += 1) {
            const next_token = decoder_family.argMaxLogit(latest_logits) catch break;
            if (decoder_family.isEosToken(text_runtime.cfg.architecture, next_token)) break;
            const next_token_u32 = std.math.cast(u32, next_token) orelse break;
            try generated_ids.append(allocator, next_token_u32);
            latest_logits = text_runtime.forwardTokenIdWithPosition(
                &workspace,
                &cache,
                next_token,
                core.textTokenPosition(text_runtime.cfg.rope_position_mode, position_plan.generation_start_position + step),
            ) catch break;
        }

        summary.decoder_logits_dim = latest_logits.len;
        summary.decoder_next_token_id = decoder_family.argMaxLogit(latest_logits) catch summary.decoder_next_token_id;
        summary.decoded_token_count = generated_ids.items.len;
        summary.text_decode_executed = generated_ids.items.len != 0;

        if (generated_ids.items.len != 0) {
            const decoded = loaded_tokenizer.decodeAlloc(allocator, generated_ids.items) catch null;
            if (decoded) |text| {
                defer allocator.free(text);
                summary.generated_output = try trimDecodedOutput(allocator, text_runtime.cfg.architecture, text);
            }
        }
    }
}

fn encodePromptTokenIds(
    allocator: std.mem.Allocator,
    operation: []const u8,
    runtime: *const decoder_runtime.Runtime,
    tokenizer: ?*decoder_family.Tokenizer,
) !?[]usize {
    const loaded_tokenizer = tokenizer orelse return null;
    const prompt_text = decoder_family.renderSingleUserPromptAlloc(
        allocator,
        runtime.cfg.architecture,
        instructionForOperation(operation),
        .disabled,
    ) catch return null;
    defer allocator.free(prompt_text);

    const encoded = loaded_tokenizer.encodeAlloc(allocator, prompt_text) catch return null;
    defer allocator.free(encoded);

    const ids = allocator.alloc(usize, encoded.len) catch return null;
    for (encoded, 0..) |token_id, index| {
        ids[index] = token_id;
    }
    return ids;
}

fn trimDecodedOutput(
    allocator: std.mem.Allocator,
    architecture: decoder_family.Architecture,
    text: []const u8,
) ![]u8 {
    const effective_stops = decoder_family.effectiveStopSequencesAlloc(
        allocator,
        architecture,
        &.{},
    ) catch null;
    if (effective_stops) |stops| {
        defer allocator.free(stops);
        const analysis = streaming.analyzeGeneratedText(text, stops);
        return try allocator.dupe(u8, text[0..analysis.response_len]);
    }
    return try allocator.dupe(u8, text);
}

fn buildOutputJson(
    allocator: std.mem.Allocator,
    context: core.Context,
    readiness: core.Readiness,
    preprocess_summary: ?core.PreprocessSummary,
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
