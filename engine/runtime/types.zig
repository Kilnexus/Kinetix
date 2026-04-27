const std = @import("std");
const backend = @import("../artifacts/backend/backend.zig");
const abi = @import("runtime_abi");
const task = @import("../core/task.zig");

pub const RuntimeAbi = abi;
pub const RuntimeAbiVersion = abi.Version;
pub const RuntimeOperation = abi.Operation;
pub const Modality = task.Modality;
pub const ExecutionMode = task.ExecutionMode;
pub const InputPayload = task.InputPayload;
pub const GenerationOptions = task.GenerationOptions;
pub const WeightScheme = backend.WeightScheme;

pub const Descriptor = struct {
    id: []const u8,
    modality: Modality,
    version: []const u8 = "0.1.0",
    bound_model_family: ?[]const u8 = null,
    supports_batching: bool = false,
    supports_streaming: bool = false,
    supported_operations: []const []const u8 = &.{},
    supported_operation_ids: []const RuntimeOperation = &.{},

    pub fn supportsOperation(self: Descriptor, operation: []const u8) bool {
        if (self.supported_operations.len == 0 and self.supported_operation_ids.len == 0) return true;
        if (self.supported_operation_ids.len != 0) {
            const operation_id = RuntimeOperation.parse(operation) orelse return false;
            return self.supportsRuntimeOperation(operation_id);
        }
        return self.supportsOperationNameFallback(operation);
    }

    pub fn supportsRuntimeOperation(self: Descriptor, operation: RuntimeOperation) bool {
        if (self.supported_operation_ids.len == 0) return self.supportsOperationNameFallback(operation.name());
        for (self.supported_operation_ids) |supported| {
            if (supported == operation) return true;
        }
        return false;
    }

    fn supportsOperationNameFallback(self: Descriptor, operation: []const u8) bool {
        if (self.supported_operations.len == 0) return true;
        for (self.supported_operations) |supported| {
            if (std.mem.eql(u8, supported, operation)) return true;
        }
        return false;
    }
};

pub const Submission = struct {
    adapter_id: []const u8,
    accepted: bool = true,
    execution: ExecutionMode,
};

pub const OutputPayload = union(enum) {
    none,
    text: []const u8,
    json: []const u8,
    audio_path: []const u8,

    pub fn deinit(self: *OutputPayload, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .none => {},
            .text => |value| allocator.free(value),
            .json => |value| allocator.free(value),
            .audio_path => |value| allocator.free(value),
        }
        self.* = .none;
    }
};

pub const ExecutionOrigin = abi.ExecutionOrigin;

pub const ExecutionNote = enum {
    none,
    validated_only,
    text_request_ready,
    text_native_qwen_single,
    text_native_qwen_batch,
    vision_graph_ready,
    vision_shared_detect,
    ocr_model_ready,
    ocr_swiftocr_native,
    ocr_chandra_native,
    tts_model_ready,
};

pub const ExecutionResult = struct {
    submission: Submission,
    origin: ExecutionOrigin = .runtime_backend,
    note: ExecutionNote = .none,
    output: OutputPayload = .none,

    pub fn deinit(self: *ExecutionResult, allocator: std.mem.Allocator) void {
        self.output.deinit(allocator);
        self.* = undefined;
    }
};

pub const BatchExecutionPath = abi.BatchExecutionPath;

pub const InputKind = abi.InputKind;

pub const ProviderKey = enum {
    qwen3_text,
    bert_text,
    yolo_vision,
    swiftocr_ocr,
    chandra_ocr,
    moss_tts_nano_tts,
    generic,

    pub fn name(self: ProviderKey) []const u8 {
        return switch (self) {
            .qwen3_text => "qwen3_text",
            .bert_text => "bert_text",
            .yolo_vision => "yolo_vision",
            .swiftocr_ocr => "swiftocr_ocr",
            .chandra_ocr => "chandra_ocr",
            .moss_tts_nano_tts => "moss_tts_nano_tts",
            .generic => "generic",
        };
    }
};

pub const SourceFormat = enum {
    huggingface_directory,
    kinetix_graph_directory,
    swiftocr_bundle,
    onnx_bundle,
    unknown,
};

pub const NormalizedFormat = enum {
    text_decoder,
    vision_graph,
    ocr_bundle,
    document_vlm,
    tts_onnx_bundle,
    generic,
};

pub const RuntimeSupportStatus = enum {
    supported,
    degraded,
    unsupported,
};

pub const RuntimeSupportWarning = enum {
    graph_runtime_backend_required,
    native_batch_unavailable,
    document_input_partial,
    tts_runtime_pending,
};

pub const RuntimeSupportRewrite = enum {
    quantized_weights_selected,
    graph_schema_accepted,
    ocr_single_file_bundle,
};

pub const Diagnostic = struct {
    code: []const u8,
    message: []const u8,
};

pub const RuntimeRequest = struct {
    operation: []const u8,
    operation_id: RuntimeOperation = .infer,
    input: InputPayload = .none,
    execution: ExecutionMode = .sync,
    generation: GenerationOptions = .{},
    allows_batching: bool = true,

    pub fn resolvedOperationId(self: RuntimeRequest) RuntimeOperation {
        return RuntimeOperation.parse(self.operation) orelse self.operation_id;
    }
};

pub const RuntimeBatchRequest = struct {
    items: []const RuntimeRequest,
};

pub const PlanBatch = struct {
    allocator: std.mem.Allocator,
    request_indices: []usize,
    operation: []const u8,
    operation_id: RuntimeOperation = .infer,
    execution: ExecutionMode,
    allows_batching: bool,

    pub fn deinit(self: *PlanBatch) void {
        self.allocator.free(self.request_indices);
        self.* = undefined;
    }
};

pub const ExecutionPath = abi.ExecutionPath;

pub const ExecutionPlan = struct {
    allocator: std.mem.Allocator,
    model_id: []const u8,
    request_count: usize,
    execution: ExecutionMode,
    path: ExecutionPath,
    requests: []RuntimeRequest,
    batches: []PlanBatch,

    pub fn deinit(self: *ExecutionPlan) void {
        self.allocator.free(self.requests);
        for (self.batches) |*batch| batch.deinit();
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

pub const RuntimeResult = struct {
    origin: ExecutionOrigin = .runtime_backend,
    note: ExecutionNote = .none,
    output: OutputPayload = .none,
    diagnostics: []const Diagnostic = &.{},

    pub fn deinit(self: *RuntimeResult, allocator: std.mem.Allocator) void {
        self.output.deinit(allocator);
        self.* = undefined;
    }
};

pub const RuntimeVisionDetection = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
};

pub const RuntimeVisionDetectOutput = struct {
    candidate_count: usize,
    detections: []RuntimeVisionDetection,

    pub fn deinit(self: *RuntimeVisionDetectOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.detections);
        self.* = undefined;
    }
};

pub const RuntimeBatchResults = struct {
    allocator: std.mem.Allocator,
    items: []RuntimeResult,

    pub fn deinit(self: *RuntimeBatchResults) void {
        for (self.items) |*item| item.deinit(self.allocator);
        self.allocator.free(self.items);
        self.* = undefined;
    }
};

pub const RequestExecutionResult = struct {
    request_index: usize,
    result: ExecutionResult,
};

pub const ExecutedBatch = struct {
    adapter_id: []const u8,
    execution: ExecutionMode,
    supports_batching: bool,
    execute_path: BatchExecutionPath,
    request_results: []RequestExecutionResult,

    pub fn len(self: ExecutedBatch) usize {
        return self.request_results.len;
    }

    pub fn acceptedCount(self: ExecutedBatch) usize {
        var accepted: usize = 0;
        for (self.request_results) |result| {
            accepted += @intFromBool(result.result.submission.accepted);
        }
        return accepted;
    }
};

pub const BatchExecutionReport = struct {
    allocator: std.mem.Allocator,
    batches: []ExecutedBatch,

    pub fn deinit(self: *BatchExecutionReport) void {
        for (self.batches) |batch| {
            for (batch.request_results) |*result| result.result.deinit(self.allocator);
            self.allocator.free(batch.request_results);
        }
        self.allocator.free(self.batches);
        self.* = undefined;
    }

    pub fn totalRequests(self: BatchExecutionReport) usize {
        var total: usize = 0;
        for (self.batches) |batch| total += batch.len();
        return total;
    }

    pub fn totalAccepted(self: BatchExecutionReport) usize {
        var total: usize = 0;
        for (self.batches) |batch| total += batch.acceptedCount();
        return total;
    }
};

pub fn inputKind(payload: InputPayload) InputKind {
    return switch (payload) {
        .none => .none,
        .text => .text,
        .image_path => .image_path,
        .document_path => .document_path,
        .audio_path => .audio_path,
        .video_path => .video_path,
    };
}

test "descriptor treats operation strings as ABI names when ids are present" {
    const descriptor = Descriptor{
        .id = "test",
        .modality = .text,
        .supported_operations = &.{"legacy-generate"},
        .supported_operation_ids = &.{.generate},
    };

    try std.testing.expect(descriptor.supportsOperation("generate"));
    try std.testing.expect(!descriptor.supportsOperation("legacy-generate"));
    try std.testing.expect(!descriptor.supportsOperation("unknown"));
}
