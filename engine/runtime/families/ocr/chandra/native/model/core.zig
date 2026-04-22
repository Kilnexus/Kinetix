const std = @import("std");
const task = @import("../../../../../../core/task.zig");
const preprocess = @import("../../preprocess.zig");
const store = @import("../../store.zig");
const weights = @import("../../weights.zig");
const decoder_types = @import("../../../../../text/decoder_types.zig");
const input = @import("../input/loader.zig");

pub const io = std.Options.debug_io;

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

pub const PreprocessSummary = struct {
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

pub const MultimodalPositionPlan = struct {
    vision_start_position: usize,
    visual_start_position: usize,
    vision_end_position: usize,
    prompt_start_position: usize,
    generation_start_position: usize,
    total_prefill_tokens: usize,

    pub fn init(
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

pub fn visualTokenPosition(
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

pub fn textTokenPosition(
    rope_position_mode: decoder_types.RopePositionMode,
    scalar_position: usize,
) decoder_types.TokenPosition {
    if (rope_position_mode == .scalar) {
        return decoder_types.TokenPosition.scalarPosition(scalar_position);
    }
    return decoder_types.TokenPosition.mropePosition(.{ scalar_position, scalar_position, scalar_position, scalar_position });
}

pub fn maxVisualPosition(
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

pub fn allocMultimodalTextPositions(
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

pub fn allocMultimodalVisualPositions(
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
        .has_config = input.hasFile(model_path, "config.json"),
        .has_tokenizer = input.hasAnyFile(model_path, &.{ "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt" }),
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
