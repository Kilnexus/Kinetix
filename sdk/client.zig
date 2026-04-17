const std = @import("std");
const engine = @import("engine_root");
const execution = @import("sdk_execution");

const backend = engine.artifacts.backend;
const runtime_types = engine.runtime.types;
const task = engine.core.task;

pub const TextGenerateOptions = struct {
    operation: []const u8 = "generate",
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const TextBatchItem = struct {
    input: []const u8,
    operation: []const u8 = "generate",
    execution: task.ExecutionMode = .sync,
    max_tokens: ?usize = null,
    native_exec: bool = false,
    allows_batching: bool = true,
};

pub const TextBatchOptions = struct {
    preferred_weights: backend.WeightScheme = .auto,
};

pub const DetectOptions = struct {
    operation: []const u8 = "detect",
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
};

pub const OCROptions = struct {
    operation: []const u8 = "infer-ocr",
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
};

pub const OpenModelOptions = struct {
    preferred_weights: backend.WeightScheme = .auto,
};

pub const TextModelGenerateOptions = struct {
    execution: task.ExecutionMode = .sync,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const TextModelBatchItem = struct {
    input: []const u8,
    execution: task.ExecutionMode = .sync,
    max_tokens: ?usize = null,
    native_exec: bool = false,
    allows_batching: bool = true,
};

pub const TextModelChatOptions = struct {
    execution: task.ExecutionMode = .sync,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const VisionModelDetectOptions = struct {
    execution: task.ExecutionMode = .sync,
};

pub const OCRModelInferOptions = struct {
    execution: task.ExecutionMode = .sync,
};

pub const TextGenerationResult = struct {
    adapter_id: []u8,
    model_family: []u8,
    accepted: bool,
    execution: task.ExecutionMode,
    origin: runtime_types.ExecutionOrigin,
    note: runtime_types.ExecutionNote,
    text: []u8,

    pub fn deinit(self: *TextGenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.adapter_id);
        allocator.free(self.model_family);
        allocator.free(self.text);
        self.* = undefined;
    }
};

pub const TextBatchGenerationItem = struct {
    accepted: bool,
    execution: task.ExecutionMode,
    origin: runtime_types.ExecutionOrigin,
    note: runtime_types.ExecutionNote,
    text: []u8,

    pub fn deinit(self: *TextBatchGenerationItem, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        self.* = undefined;
    }
};

pub const TextBatchGenerationResult = struct {
    adapter_id: []u8,
    model_family: []u8,
    used_batching: bool,
    items: []TextBatchGenerationItem,

    pub fn deinit(self: *TextBatchGenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.adapter_id);
        allocator.free(self.model_family);
        for (self.items) |*item| item.deinit(allocator);
        allocator.free(self.items);
        self.* = undefined;
    }
};

pub const Detection = struct {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    score: f64,
    class_id: usize,
};

pub const DetectionResult = struct {
    adapter_id: []u8,
    accepted: bool,
    execution: task.ExecutionMode,
    origin: runtime_types.ExecutionOrigin,
    note: runtime_types.ExecutionNote,
    status: []u8,
    operation: []u8,
    model_name: []u8,
    model_family: []u8,
    input_path: ?[]u8,
    execution_nodes: usize,
    tensor_count: usize,
    class_count: ?usize,
    candidate_count: ?usize,
    detections: []Detection,

    pub fn deinit(self: *DetectionResult, allocator: std.mem.Allocator) void {
        allocator.free(self.adapter_id);
        allocator.free(self.status);
        allocator.free(self.operation);
        allocator.free(self.model_name);
        allocator.free(self.model_family);
        if (self.input_path) |value| allocator.free(value);
        allocator.free(self.detections);
        self.* = undefined;
    }
};

pub const OCRResult = struct {
    adapter_id: []u8,
    accepted: bool,
    execution: task.ExecutionMode,
    origin: runtime_types.ExecutionOrigin,
    note: runtime_types.ExecutionNote,
    status: []u8,
    operation: []u8,
    model_family: []u8,
    model_path: []u8,
    input_path: ?[]u8,
    loaded_tensors: ?usize,
    image_width: ?usize,
    image_height: ?usize,
    resized_width: ?usize,
    resized_height: ?usize,
    patch_token_count: ?usize,
    visual_token_count: ?usize,
    patch_embedding_dim: ?usize,
    patch_embedding_executed: bool,
    visual_attention_dim: ?usize,
    visual_block_attention_executed: bool,
    visual_block_dim: ?usize,
    visual_block_mlp_executed: bool,
    visual_token_dim: ?usize,
    visual_merger_executed: bool,
    backend: ?[]u8,
    method: ?[]u8,
    requested_output: ?[]u8,
    content: ?[]u8,
    markdown: ?[]u8,
    html: ?[]u8,
    json_output: ?[]u8,
    page_count: ?usize,
    total_token_count: ?usize,
    error_message: ?[]u8,

    pub fn deinit(self: *OCRResult, allocator: std.mem.Allocator) void {
        allocator.free(self.adapter_id);
        allocator.free(self.status);
        allocator.free(self.operation);
        allocator.free(self.model_family);
        allocator.free(self.model_path);
        if (self.input_path) |value| allocator.free(value);
        if (self.backend) |value| allocator.free(value);
        if (self.method) |value| allocator.free(value);
        if (self.requested_output) |value| allocator.free(value);
        if (self.content) |value| allocator.free(value);
        if (self.markdown) |value| allocator.free(value);
        if (self.html) |value| allocator.free(value);
        if (self.json_output) |value| allocator.free(value);
        if (self.error_message) |value| allocator.free(value);
        self.* = undefined;
    }
};

const ParsedVisionReceipt = struct {
    status: []const u8,
    operation: []const u8,
    model_name: []const u8,
    model_family: []const u8,
    input_path: ?[]const u8,
    execution_nodes: usize,
    tensor_count: usize,
    class_count: ?usize,
    candidate_count: ?usize,
    detections: []Detection,
};

const ParsedOCRReceipt = struct {
    status: []const u8,
    operation: []const u8,
    model_family: []const u8,
    model_path: []const u8,
    input_path: ?[]const u8,
    loaded_tensors: ?usize,
    image_width: ?usize,
    image_height: ?usize,
    resized_width: ?usize = null,
    resized_height: ?usize = null,
    patch_token_count: ?usize = null,
    visual_token_count: ?usize = null,
    patch_embedding_dim: ?usize = null,
    patch_embedding_executed: bool = false,
    visual_attention_dim: ?usize = null,
    visual_block_attention_executed: bool = false,
    visual_block_dim: ?usize = null,
    visual_block_mlp_executed: bool = false,
    visual_token_dim: ?usize = null,
    visual_merger_executed: bool = false,
    backend: ?[]const u8 = null,
    method: ?[]const u8 = null,
    requested_output: ?[]const u8 = null,
    content: ?[]const u8 = null,
    markdown: ?[]const u8 = null,
    html: ?[]const u8 = null,
    json_output: ?[]const u8 = null,
    page_count: ?usize = null,
    total_token_count: ?usize = null,
    error_message: ?[]const u8 = null,
};

pub const TextModel = struct {
    context: *execution.ExecutionContext,

    pub fn deinit(self: *TextModel) void {
        destroyExecutionContext(self.context);
        self.* = undefined;
    }

    pub fn generate(self: *const TextModel, prompt: []const u8, options: TextModelGenerateOptions) !TextGenerationResult {
        var result = try self.context.execute(.{
            .operation = "generate",
            .input = prompt,
            .execution = options.execution,
            .max_tokens = options.max_tokens,
            .native_exec = options.native_exec,
        });
        defer result.deinit(self.context.allocator);

        return try copyTextGenerationResultFromDescriptor(self.context.allocator, self.context.descriptor, result);
    }

    pub fn chat(self: *const TextModel, prompt: []const u8, options: TextModelChatOptions) !TextGenerationResult {
        var result = try self.context.execute(.{
            .operation = "chat",
            .input = prompt,
            .execution = options.execution,
            .max_tokens = options.max_tokens,
            .native_exec = options.native_exec,
        });
        defer result.deinit(self.context.allocator);

        return try copyTextGenerationResultFromDescriptor(self.context.allocator, self.context.descriptor, result);
    }

    pub fn generateBatch(self: *const TextModel, items: []const TextModelBatchItem) !TextBatchGenerationResult {
        if (items.len == 0) return error.EmptyBatch;

        const batch_items = try self.context.allocator.alloc(execution.ContextBatchItem, items.len);
        defer self.context.allocator.free(batch_items);

        for (items, batch_items) |item, *slot| {
            slot.* = .{
                .operation = "generate",
                .input = item.input,
                .execution = item.execution,
                .max_tokens = item.max_tokens,
                .native_exec = item.native_exec,
                .allows_batching = item.allows_batching,
            };
        }

        var report = self.context.executeBatch(self.context.allocator, .{ .items = batch_items }) catch {
            return try generateTextBatchSequentialFallbackForContext(self.context, items);
        };
        defer report.deinit();

        if (allBatchResultsHaveText(report)) {
            return try copyTextBatchResultFromDescriptor(self.context.allocator, self.context.descriptor, items.len, true, report);
        }

        return try generateTextBatchSequentialFallbackForContext(self.context, items);
    }
};

pub const VisionModel = struct {
    context: *execution.ExecutionContext,

    pub fn deinit(self: *VisionModel) void {
        destroyExecutionContext(self.context);
        self.* = undefined;
    }

    pub fn detect(self: *const VisionModel, image_path: []const u8, options: VisionModelDetectOptions) !DetectionResult {
        var result = try self.context.execute(.{
            .operation = "detect",
            .input = image_path,
            .execution = options.execution,
        });
        defer result.deinit(self.context.allocator);

        return try copyDetectionResult(self.context.allocator, result);
    }
};

pub const OCRModel = struct {
    context: *execution.ExecutionContext,

    pub fn deinit(self: *OCRModel) void {
        destroyExecutionContext(self.context);
        self.* = undefined;
    }

    pub fn infer(self: *const OCRModel, image_path: []const u8, options: OCRModelInferOptions) !OCRResult {
        var result = try self.context.execute(.{
            .operation = "infer-ocr",
            .input = image_path,
            .execution = options.execution,
        });
        defer result.deinit(self.context.allocator);

        return try copyOCRResult(self.context.allocator, result);
    }
};

pub const OpenedModel = union(enum) {
    text: TextModel,
    vision: VisionModel,
    ocr: OCRModel,

    pub fn deinit(self: *OpenedModel) void {
        switch (self.*) {
            .text => |*model| model.deinit(),
            .vision => |*model| model.deinit(),
            .ocr => |*model| model.deinit(),
        }
    }
};

pub const KinetixClient = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) KinetixClient {
        return .{ .allocator = allocator };
    }

    pub fn openTextModel(
        self: KinetixClient,
        model_dir: []const u8,
        options: OpenModelOptions,
    ) !TextModel {
        const context = try execution.openContext(self.allocator, .{
            .model_dir = model_dir,
            .preferred_weights = options.preferred_weights,
        });
        errdefer destroyExecutionContext(context);

        if (context.descriptor.modality != .text) return error.ModelModalityMismatch;
        return .{ .context = context };
    }

    pub fn openVisionModel(
        self: KinetixClient,
        model_dir: []const u8,
        options: OpenModelOptions,
    ) !VisionModel {
        const context = try execution.openContext(self.allocator, .{
            .model_dir = model_dir,
            .preferred_weights = options.preferred_weights,
        });
        errdefer destroyExecutionContext(context);

        if (context.descriptor.modality != .vision) return error.ModelModalityMismatch;
        return .{ .context = context };
    }

    pub fn openOCRModel(
        self: KinetixClient,
        model_dir: []const u8,
        options: OpenModelOptions,
    ) !OCRModel {
        const context = try execution.openContext(self.allocator, .{
            .model_dir = model_dir,
            .preferred_weights = options.preferred_weights,
        });
        errdefer destroyExecutionContext(context);

        if (context.descriptor.modality != .ocr) return error.ModelModalityMismatch;
        return .{ .context = context };
    }

    pub fn openModel(
        self: KinetixClient,
        model_dir: []const u8,
        options: OpenModelOptions,
    ) !OpenedModel {
        const context = try execution.openContext(self.allocator, .{
            .model_dir = model_dir,
            .preferred_weights = options.preferred_weights,
        });
        errdefer destroyExecutionContext(context);

        return switch (context.descriptor.modality) {
            .text => .{ .text = .{ .context = context } },
            .vision => .{ .vision = .{ .context = context } },
            .ocr => .{ .ocr = .{ .context = context } },
            else => return error.UnsupportedModelModality,
        };
    }

    pub fn generateText(
        self: KinetixClient,
        model_dir: []const u8,
        prompt: []const u8,
        options: TextGenerateOptions,
    ) !TextGenerationResult {
        var prepared = try execution.prepare(self.allocator, .{
            .model_dir = model_dir,
            .operation = options.operation,
            .input = prompt,
            .execution = options.execution,
            .preferred_weights = options.preferred_weights,
            .max_tokens = options.max_tokens,
            .native_exec = options.native_exec,
        });
        defer prepared.deinit();

        var result = try prepared.execute();
        defer result.deinit(self.allocator);

        return try copyTextGenerationResult(self.allocator, &prepared, result);
    }

    pub fn generateTextBatch(
        self: KinetixClient,
        model_dir: []const u8,
        items: []const TextBatchItem,
        options: TextBatchOptions,
    ) !TextBatchGenerationResult {
        if (items.len == 0) return error.EmptyBatch;

        const batch_items = try self.allocator.alloc(execution.PrepareBatchItem, items.len);
        defer self.allocator.free(batch_items);

        for (items, batch_items) |item, *slot| {
            slot.* = .{
                .operation = item.operation,
                .input = item.input,
                .execution = item.execution,
                .max_tokens = item.max_tokens,
                .native_exec = item.native_exec,
                .allows_batching = item.allows_batching,
            };
        }

        var prepared = try execution.prepareBatch(self.allocator, .{
            .model_dir = model_dir,
            .preferred_weights = options.preferred_weights,
            .items = batch_items,
        });
        defer prepared.deinit();

        var report = prepared.execute() catch {
            return try generateTextBatchSequentialFallback(self, model_dir, items, options.preferred_weights);
        };
        defer report.deinit();

        if (allBatchResultsHaveText(report)) {
            return try copyTextBatchResult(self.allocator, &prepared, report);
        }

        return try generateTextBatchSequentialFallback(self, model_dir, items, options.preferred_weights);
    }

    pub fn detect(
        self: KinetixClient,
        model_dir: []const u8,
        image_path: []const u8,
        options: DetectOptions,
    ) !DetectionResult {
        var prepared = try execution.prepare(self.allocator, .{
            .model_dir = model_dir,
            .operation = options.operation,
            .input = image_path,
            .execution = options.execution,
            .preferred_weights = options.preferred_weights,
        });
        defer prepared.deinit();

        var result = try prepared.execute();
        defer result.deinit(self.allocator);

        return try copyDetectionResult(self.allocator, result);
    }

    pub fn inferOCR(
        self: KinetixClient,
        model_dir: []const u8,
        image_path: []const u8,
        options: OCROptions,
    ) !OCRResult {
        var prepared = try execution.prepare(self.allocator, .{
            .model_dir = model_dir,
            .operation = options.operation,
            .input = image_path,
            .execution = options.execution,
            .preferred_weights = options.preferred_weights,
        });
        defer prepared.deinit();

        var result = try prepared.execute();
        defer result.deinit(self.allocator);

        return try copyOCRResult(self.allocator, result);
    }
};

fn copyTextGenerationResult(
    allocator: std.mem.Allocator,
    prepared: *const execution.PreparedExecution,
    result: runtime_types.ExecutionResult,
) !TextGenerationResult {
    return try copyTextGenerationResultFromDescriptor(allocator, prepared.descriptor, result);
}

fn copyTextGenerationResultFromDescriptor(
    allocator: std.mem.Allocator,
    descriptor: runtime_types.Descriptor,
    result: runtime_types.ExecutionResult,
) !TextGenerationResult {
    const text = switch (result.output) {
        .text => |value| value,
        else => return error.ExpectedTextOutput,
    };

    return .{
        .adapter_id = try allocator.dupe(u8, result.submission.adapter_id),
        .model_family = try allocator.dupe(u8, descriptor.bound_model_family orelse "unknown"),
        .accepted = result.submission.accepted,
        .execution = result.submission.execution,
        .origin = result.origin,
        .note = result.note,
        .text = try allocator.dupe(u8, text),
    };
}

fn copyTextBatchResult(
    allocator: std.mem.Allocator,
    prepared: *const execution.PreparedBatchExecution,
    report: runtime_types.BatchExecutionReport,
) !TextBatchGenerationResult {
    return try copyTextBatchResultFromDescriptor(allocator, prepared.descriptor, prepared.requests.len, true, report);
}

fn copyTextBatchResultFromDescriptor(
    allocator: std.mem.Allocator,
    descriptor: runtime_types.Descriptor,
    item_count: usize,
    used_batching: bool,
    report: runtime_types.BatchExecutionReport,
) !TextBatchGenerationResult {
    const items = try allocator.alloc(TextBatchGenerationItem, item_count);
    errdefer allocator.free(items);

    var seen = try allocator.alloc(bool, item_count);
    defer allocator.free(seen);
    @memset(seen, false);

    errdefer {
        for (items, seen) |*item, was_seen| {
            if (was_seen) item.deinit(allocator);
        }
        allocator.free(items);
    }

    for (report.batches) |batch| {
        for (batch.request_results) |request_result| {
            if (request_result.request_index >= items.len) return error.InvalidRequestIndex;
            if (seen[request_result.request_index]) return error.DuplicateBatchResult;

            items[request_result.request_index] = try copyTextBatchItem(allocator, request_result.result);
            seen[request_result.request_index] = true;
        }
    }

    for (seen) |was_seen| {
        if (!was_seen) return error.MissingBatchResult;
    }

    return .{
        .adapter_id = try allocator.dupe(u8, descriptor.id),
        .model_family = try allocator.dupe(u8, descriptor.bound_model_family orelse "unknown"),
        .used_batching = used_batching,
        .items = items,
    };
}

fn copyTextBatchItem(allocator: std.mem.Allocator, result: runtime_types.ExecutionResult) !TextBatchGenerationItem {
    const text = switch (result.output) {
        .text => |value| value,
        else => return error.ExpectedTextOutput,
    };

    return .{
        .accepted = result.submission.accepted,
        .execution = result.submission.execution,
        .origin = result.origin,
        .note = result.note,
        .text = try allocator.dupe(u8, text),
    };
}

fn copyDetectionResult(allocator: std.mem.Allocator, result: runtime_types.ExecutionResult) !DetectionResult {
    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };

    const parsed = try std.json.parseFromSlice(ParsedVisionReceipt, allocator, payload, .{});
    defer parsed.deinit();

    const adapter_id = try allocator.dupe(u8, result.submission.adapter_id);
    errdefer allocator.free(adapter_id);
    const status = try allocator.dupe(u8, parsed.value.status);
    errdefer allocator.free(status);
    const operation = try allocator.dupe(u8, parsed.value.operation);
    errdefer allocator.free(operation);
    const model_name = try allocator.dupe(u8, parsed.value.model_name);
    errdefer allocator.free(model_name);
    const model_family = try allocator.dupe(u8, parsed.value.model_family);
    errdefer allocator.free(model_family);
    const input_path = try dupeOptionalString(allocator, parsed.value.input_path);
    errdefer if (input_path) |value| allocator.free(value);
    const detections = try copyDetections(allocator, parsed.value.detections);
    errdefer allocator.free(detections);

    return .{
        .adapter_id = adapter_id,
        .accepted = result.submission.accepted,
        .execution = result.submission.execution,
        .origin = result.origin,
        .note = result.note,
        .status = status,
        .operation = operation,
        .model_name = model_name,
        .model_family = model_family,
        .input_path = input_path,
        .execution_nodes = parsed.value.execution_nodes,
        .tensor_count = parsed.value.tensor_count,
        .class_count = parsed.value.class_count,
        .candidate_count = parsed.value.candidate_count,
        .detections = detections,
    };
}

fn copyOCRResult(allocator: std.mem.Allocator, result: runtime_types.ExecutionResult) !OCRResult {
    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };

    const parsed = try std.json.parseFromSlice(ParsedOCRReceipt, allocator, payload, .{});
    defer parsed.deinit();

    const adapter_id = try allocator.dupe(u8, result.submission.adapter_id);
    errdefer allocator.free(adapter_id);
    const status = try allocator.dupe(u8, parsed.value.status);
    errdefer allocator.free(status);
    const operation = try allocator.dupe(u8, parsed.value.operation);
    errdefer allocator.free(operation);
    const model_family = try allocator.dupe(u8, parsed.value.model_family);
    errdefer allocator.free(model_family);
    const model_path = try allocator.dupe(u8, parsed.value.model_path);
    errdefer allocator.free(model_path);
    const input_path = try dupeOptionalString(allocator, parsed.value.input_path);
    errdefer if (input_path) |value| allocator.free(value);

    return .{
        .adapter_id = adapter_id,
        .accepted = result.submission.accepted,
        .execution = result.submission.execution,
        .origin = result.origin,
        .note = result.note,
        .status = status,
        .operation = operation,
        .model_family = model_family,
        .model_path = model_path,
        .input_path = input_path,
        .loaded_tensors = parsed.value.loaded_tensors,
        .image_width = parsed.value.image_width,
        .image_height = parsed.value.image_height,
        .resized_width = parsed.value.resized_width,
        .resized_height = parsed.value.resized_height,
        .patch_token_count = parsed.value.patch_token_count,
        .visual_token_count = parsed.value.visual_token_count,
        .patch_embedding_dim = parsed.value.patch_embedding_dim,
        .patch_embedding_executed = parsed.value.patch_embedding_executed,
        .visual_attention_dim = parsed.value.visual_attention_dim,
        .visual_block_attention_executed = parsed.value.visual_block_attention_executed,
        .visual_block_dim = parsed.value.visual_block_dim,
        .visual_block_mlp_executed = parsed.value.visual_block_mlp_executed,
        .visual_token_dim = parsed.value.visual_token_dim,
        .visual_merger_executed = parsed.value.visual_merger_executed,
        .backend = try dupeOptionalString(allocator, parsed.value.backend),
        .method = try dupeOptionalString(allocator, parsed.value.method),
        .requested_output = try dupeOptionalString(allocator, parsed.value.requested_output),
        .content = try dupeOptionalString(allocator, parsed.value.content),
        .markdown = try dupeOptionalString(allocator, parsed.value.markdown),
        .html = try dupeOptionalString(allocator, parsed.value.html),
        .json_output = try dupeOptionalString(allocator, parsed.value.json_output),
        .page_count = parsed.value.page_count,
        .total_token_count = parsed.value.total_token_count,
        .error_message = try dupeOptionalString(allocator, parsed.value.error_message),
    };
}

fn copyDetections(allocator: std.mem.Allocator, detections: []const Detection) ![]Detection {
    const owned = try allocator.alloc(Detection, detections.len);
    for (detections, owned) |src, *dst| dst.* = src;
    return owned;
}

fn dupeOptionalString(allocator: std.mem.Allocator, value: ?[]const u8) !?[]u8 {
    if (value) |string| return try allocator.dupe(u8, string);
    return null;
}

fn destroyExecutionContext(context: *execution.ExecutionContext) void {
    const allocator = context.allocator;
    context.deinit();
    allocator.destroy(context);
}

fn allBatchResultsHaveText(report: runtime_types.BatchExecutionReport) bool {
    for (report.batches) |batch| {
        for (batch.request_results) |request_result| {
            switch (request_result.result.output) {
                .text => {},
                else => return false,
            }
        }
    }
    return true;
}

fn generateTextBatchSequentialFallback(
    self: KinetixClient,
    model_dir: []const u8,
    items: []const TextBatchItem,
    preferred_weights: backend.WeightScheme,
) !TextBatchGenerationResult {
    const results = try self.allocator.alloc(TextBatchGenerationItem, items.len);
    errdefer self.allocator.free(results);

    const seen = try self.allocator.alloc(bool, items.len);
    defer self.allocator.free(seen);
    @memset(seen, false);

    errdefer {
        for (results, seen) |*item, was_seen| {
            if (was_seen) item.deinit(self.allocator);
        }
        self.allocator.free(results);
    }

    var adapter_id: ?[]u8 = null;
    errdefer if (adapter_id) |value| self.allocator.free(value);
    var model_family: ?[]u8 = null;
    errdefer if (model_family) |value| self.allocator.free(value);

    for (items, 0..) |item, index| {
        var single = try self.generateText(model_dir, item.input, .{
            .operation = item.operation,
            .execution = item.execution,
            .preferred_weights = preferred_weights,
            .max_tokens = item.max_tokens,
            .native_exec = item.native_exec,
        });
        defer single.deinit(self.allocator);

        if (adapter_id == null) adapter_id = try self.allocator.dupe(u8, single.adapter_id);
        if (model_family == null) model_family = try self.allocator.dupe(u8, single.model_family);

        results[index] = .{
            .accepted = single.accepted,
            .execution = single.execution,
            .origin = single.origin,
            .note = single.note,
            .text = try self.allocator.dupe(u8, single.text),
        };
        seen[index] = true;
    }

    return .{
        .adapter_id = adapter_id.?,
        .model_family = model_family.?,
        .used_batching = false,
        .items = results,
    };
}

fn generateTextBatchSequentialFallbackForContext(
    context: *const execution.ExecutionContext,
    items: []const TextModelBatchItem,
) !TextBatchGenerationResult {
    const results = try context.allocator.alloc(TextBatchGenerationItem, items.len);
    errdefer context.allocator.free(results);

    const seen = try context.allocator.alloc(bool, items.len);
    defer context.allocator.free(seen);
    @memset(seen, false);

    errdefer {
        for (results, seen) |*item, was_seen| {
            if (was_seen) item.deinit(context.allocator);
        }
        context.allocator.free(results);
    }

    for (items, 0..) |item, index| {
        var single = try context.execute(.{
            .operation = "generate",
            .input = item.input,
            .execution = item.execution,
            .max_tokens = item.max_tokens,
            .native_exec = item.native_exec,
        });
        defer single.deinit(context.allocator);

        results[index] = try copyTextBatchItem(context.allocator, single);
        seen[index] = true;
    }

    return .{
        .adapter_id = try context.allocator.dupe(u8, context.descriptor.id),
        .model_family = try context.allocator.dupe(u8, context.descriptor.bound_model_family orelse "unknown"),
        .used_batching = false,
        .items = results,
    };
}

test "client generateText returns typed result for qwen3 native execution" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var result = try client.generateText(root_path, "hello", .{
        .native_exec = true,
        .max_tokens = 8,
    });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("qwen3", result.model_family);
    try std.testing.expectEqual(runtime_types.ExecutionOrigin.native_single_bridge, result.origin);
    try std.testing.expectEqual(runtime_types.ExecutionNote.text_native_qwen_single, result.note);
    try std.testing.expectEqualStrings("stub-native-single", result.text);
}

test "client detect returns typed detection result" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json",
        \\{
        \\  "format_version": 1,
        \\  "model_name": "vision-yolo",
        \\  "metadata": {},
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "pipeline.detect", "kind": "Detect", "from": [-1] }
        \\  ],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": []
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var result = try client.detect(root_path, "demo.png", .{});
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("detect_completed", result.status);
    try std.testing.expectEqualStrings("yolo", result.model_family);
    try std.testing.expectEqual(@as(?usize, 4), result.candidate_count);
    try std.testing.expectEqual(@as(usize, 1), result.detections.len);
    try std.testing.expectEqual(@as(usize, 1), result.detections[0].class_id);
}

test "client inferOCR returns typed result" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeOCRModel(tmp.dir, "demo.swm", 0);
    try writePPMImage(tmp.dir, "demo.ppm", 1, 1, &[_]u8{ 1, 2, 3 });

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.ppm");
    defer std.testing.allocator.free(image_path);

    const client = KinetixClient.init(std.testing.allocator);
    var result = try client.inferOCR(root_path, image_path, .{});
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("ocr_infer_completed", result.status);
    try std.testing.expectEqualStrings("swiftocr", result.model_family);
    try std.testing.expectEqual(@as(?usize, 0), result.loaded_tensors);
    try std.testing.expectEqual(@as(?usize, 1), result.image_width);
    try std.testing.expectEqual(@as(?usize, 1), result.image_height);
}

test "client generateTextBatch returns batched typed generation when unified runtime materializes outputs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const requests = [_]TextBatchItem{
        .{ .input = "hello", .native_exec = true, .max_tokens = 8 },
        .{ .input = "world", .native_exec = true, .max_tokens = 8 },
    };

    const client = KinetixClient.init(std.testing.allocator);
    var result = try client.generateTextBatch(root_path, &requests, .{});
    defer result.deinit(std.testing.allocator);

    try std.testing.expect(result.used_batching);
    try std.testing.expectEqual(@as(usize, 2), result.items.len);
    try std.testing.expectEqualStrings("stub-native-batch", result.items[0].text);
    try std.testing.expectEqualStrings("stub-native-batch", result.items[1].text);
}

test "client openTextModel exposes operation-specific methods without operation strings" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var model = try client.openTextModel(root_path, .{});
    defer model.deinit();

    var generate_result = try model.generate("hello", .{
        .native_exec = true,
        .max_tokens = 8,
    });
    defer generate_result.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("stub-native-single", generate_result.text);

    var chat_result = try model.chat("hello", .{
        .native_exec = true,
        .max_tokens = 8,
    });
    defer chat_result.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("stub-native-single", chat_result.text);
}

test "client openModel auto-detects text models" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var opened = try client.openModel(root_path, .{});
    defer opened.deinit();

    switch (opened) {
        .text => {},
        else => return error.ExpectedTextModel,
    }
}

test "client openModel auto-detects vision models" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json",
        \\{
        \\  "format_version": 1,
        \\  "model_name": "vision-yolo",
        \\  "metadata": {},
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "pipeline.detect", "kind": "Detect", "from": [-1] }
        \\  ],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": []
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var opened = try client.openModel(root_path, .{});
    defer opened.deinit();

    switch (opened) {
        .vision => {},
        else => return error.ExpectedVisionModel,
    }
}

test "client openModel auto-detects ocr models" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeOCRModel(tmp.dir, "demo.swm", 0);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const client = KinetixClient.init(std.testing.allocator);
    var opened = try client.openModel(root_path, .{});
    defer opened.deinit();

    switch (opened) {
        .ocr => {},
        else => return error.ExpectedOCRModel,
    }
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn writeOCRModel(dir: std.fs.Dir, relative_path: []const u8, tensor_count: u32) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();

    var writer_impl = file.writer(&.{});
    const writer = &writer_impl.interface;
    try writer.writeAll(&[_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 });
    try writer.writeInt(u32, tensor_count, .little);
    try writer.flush();
}

fn writePPMImage(dir: std.fs.Dir, relative_path: []const u8, width: usize, height: usize, pixels: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();

    var writer_impl = file.writer(&.{});
    const writer = &writer_impl.interface;
    try writer.print("P6\n{d} {d}\n255\n", .{ width, height });
    try writer.writeAll(pixels);
    try writer.flush();
}
