const std = @import("std");
const imaging = @import("Pixio");
const task = @import("../../core/task.zig");
const preprocess = @import("chandra_preprocess.zig");
const store = @import("chandra_store.zig");
const vision = @import("chandra_vision.zig");
const weights = @import("chandra_weights.zig");

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
};

pub fn execute(allocator: std.mem.Allocator, context: Context) ![]u8 {
    if (context.execution != .sync) return error.UnsupportedExecutionMode;
    const readiness = inspect(context.model_path);
    var summary: ?PreprocessSummary = null;

    if (readiness.has_image_processor_config and isRasterImagePath(context.input_path)) {
        var image_processor = try preprocess.loadImageProcessorConfig(allocator, context.model_path);
        defer image_processor.deinit();

        var prepared = try preprocess.loadImageInput(allocator, context.input_path, image_processor.value);
        defer prepared.deinit();

        var parsed_config: ?ParsedConfig = null;
        const config_path = std.fs.path.join(allocator, &.{ context.model_path, "config.json" }) catch null;
        defer if (config_path) |path| allocator.free(path);
        if (config_path) |path| {
            parsed_config = loadConfigFromFile(allocator, path) catch null;
        }
        defer if (parsed_config) |*config| config.deinit();

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
        if (readiness.has_patch_embedding_weight) {
            var tensor_store = store.ChandraStore.open(allocator, context.model_path) catch null;
            if (tensor_store) |*opened| {
                defer opened.deinit();
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

                        var grouped = vision.mergeSpatialPatches(allocator, merged_input, image_processor.value.merge_size) catch null;
                        if (grouped) |*merged| {
                            defer merged.deinit();
                            var merger_weights = opened.loadVisualMergerWeights(allocator) catch null;
                            if (merger_weights) |*projector| {
                                defer projector.deinit();
                                var visual_tokens = vision.applyVisualMerger(allocator, merged.*, projector.weights) catch null;
                                if (visual_tokens) |*tokens| {
                                    visual_token_dim = tokens.embedding_dim;
                                    visual_merger_executed = true;
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
        };
    }

    return try buildIncompleteOutputJson(allocator, context, readiness, summary);
}

pub fn inspect(model_path: []const u8) Readiness {
    var supported_config = false;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
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
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 2 * 1024 * 1024);
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
        error_message: []const u8,
        readiness: ReadinessReceipt,
    };

    const receipt = Receipt{
        .status = "ocr_native_backend_incomplete",
        .operation = context.operation,
        .model_family = "chandra",
        .model_path = context.model_path,
        .input_path = context.input_path,
        .backend = "kinetix_native",
        .method = "native",
        .requested_output = requestedOutput(context.operation),
        .native_stage = if (preprocess_summary) |summary|
            if (summary.visual_merger_executed)
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
        .content = null,
        .markdown = null,
        .html = null,
        .json_output = null,
        .page_count = if (preprocess_summary != null) 1 else null,
        .total_token_count = null,
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
        .error_message = "Chandra native inference is not complete yet; model config, weight manifest, and preprocessing readiness are available.",
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

fn hasAnyFile(model_path: []const u8, names: []const []const u8) bool {
    for (names) |name| {
        if (hasFile(model_path, name)) return true;
    }
    return false;
}

fn hasFile(model_path: []const u8, name: []const u8) bool {
    var dir = std.fs.openDirAbsolute(model_path, .{}) catch return false;
    defer dir.close();

    dir.access(name, .{}) catch return false;
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

    const config_path = try tmp.dir.realpathAlloc(std.testing.allocator, "config.json");
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

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
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
    try tmp.dir.writeFile(.{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "input.png");
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

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn writeSyntheticPatchEmbeddingSafetensors(dir: std.fs.Dir, relative_path: []const u8) !void {
    const header =
        \\{"visual.patch_embed.proj.weight":{"dtype":"F32","shape":[2,3,1,2,2],"data_offsets":[0,96]},"visual.patch_embed.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[96,104]},"visual.pos_embed":{"dtype":"F32","shape":[4,2],"data_offsets":[104,136]},"visual.blocks.0.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[136,144]},"visual.blocks.0.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[144,152]},"visual.blocks.0.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[152,200]},"visual.blocks.0.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[200,224]},"visual.blocks.0.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[224,240]},"visual.blocks.0.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[240,248]},"visual.blocks.0.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[248,256]},"visual.blocks.0.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[256,264]},"visual.blocks.0.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[264,288]},"visual.blocks.0.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[288,300]},"visual.blocks.0.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[300,324]},"visual.blocks.0.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[324,332]},"visual.blocks.1.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[332,340]},"visual.blocks.1.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[340,348]},"visual.blocks.1.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[348,396]},"visual.blocks.1.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[396,420]},"visual.blocks.1.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[420,436]},"visual.blocks.1.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[436,444]},"visual.blocks.1.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[444,452]},"visual.blocks.1.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[452,460]},"visual.blocks.1.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[460,484]},"visual.blocks.1.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[484,496]},"visual.blocks.1.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[496,520]},"visual.blocks.1.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[520,528]},"visual.merger.mlp.0.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[528,552]},"visual.merger.mlp.0.bias":{"dtype":"F32","shape":[3],"data_offsets":[552,564]},"visual.merger.mlp.2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[564,588]},"visual.merger.mlp.2.bias":{"dtype":"F32","shape":[2],"data_offsets":[588,596]},"model.embed_tokens.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[596,604]},"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[604,612]}}
    ;

    const file = try dir.createFile(relative_path, .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

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

    try file.writeAll(&payload);
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
