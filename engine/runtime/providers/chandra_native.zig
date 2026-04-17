const std = @import("std");
const task = @import("../../core/task.zig");
const preprocess = @import("chandra_preprocess.zig");
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
    visual_token_count: usize,
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

        summary = .{
            .image_width = prepared.grid.input_width,
            .image_height = prepared.grid.input_height,
            .resized_width = prepared.grid.resized_width,
            .resized_height = prepared.grid.resized_height,
            .visual_token_count = prepared.grid.token_count,
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
    var manifest = weights.loadManifest(allocator, model_path) catch null;
    if (manifest) |*loaded| {
        has_weights = loaded.len() != 0;
        weight_counts = loaded.counts;
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
        visual_token_count: ?usize,
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
        .native_stage = if (preprocess_summary != null) "image_preprocessing" else "model_loading",
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
        .visual_token_count = if (preprocess_summary) |summary| summary.visual_token_count else null,
        .error_message = "Chandra native inference is not complete yet; model config, weight manifest, and preprocessing readiness are available.",
        .readiness = .{
            .has_config = readiness.has_config,
            .has_tokenizer = readiness.has_tokenizer,
            .has_weights = readiness.has_weights,
            .has_supported_config = readiness.has_supported_config,
            .has_visual_encoder = readiness.has_visual_encoder,
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
    try std.testing.expect(readiness.has_multimodal_projector);
    try std.testing.expect(readiness.has_image_processor_config);
    try std.testing.expect(readiness.has_document_preprocessor);
    try std.testing.expectEqual(@as(usize, 1), readiness.text_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.vision_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.projector_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.output_tensor_count);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
