const std = @import("std");
const adapter_mod = @import("../adapter/adapter.zig");
const backend = @import("../artifacts/backend/backend.zig");
const task = @import("../core/task.zig");

pub const Modality = task.Modality;
pub const ExecutionMode = task.ExecutionMode;
pub const InputPayload = task.InputPayload;
pub const GenerationOptions = task.GenerationOptions;
pub const WeightScheme = backend.WeightScheme;
pub const OutputPayload = adapter_mod.OutputPayload;
pub const ExecutionOrigin = adapter_mod.ExecutionOrigin;

pub const InputKind = enum {
    none,
    text,
    image_path,
    audio_path,
    video_path,
};

pub const ProviderKey = enum {
    qwen3_text,
    bert_text,
    yolo_vision,
    swiftocr_ocr,
    generic,

    pub fn name(self: ProviderKey) []const u8 {
        return switch (self) {
            .qwen3_text => "qwen3_text",
            .bert_text => "bert_text",
            .yolo_vision => "yolo_vision",
            .swiftocr_ocr => "swiftocr_ocr",
            .generic => "generic",
        };
    }
};

pub const SourceFormat = enum {
    huggingface_directory,
    kinetix_graph_directory,
    swiftocr_bundle,
    unknown,
};

pub const NormalizedFormat = enum {
    text_decoder,
    vision_graph,
    ocr_bundle,
    generic,
};

pub const CompatibilityStatus = enum {
    supported,
    degraded,
    unsupported,
};

pub const CompatibilityWarning = enum {
    legacy_graph_bridge_required,
    native_batch_unavailable,
    ocr_pipeline_skeleton,
};

pub const CompatibilityRewrite = enum {
    quantized_weights_selected,
    legacy_graph_schema_accepted,
    ocr_single_file_bundle,
};

pub const Diagnostic = struct {
    code: []const u8,
    message: []const u8,
};

pub const RuntimeRequest = struct {
    operation: []const u8,
    input: InputPayload = .none,
    execution: ExecutionMode = .sync,
    generation: GenerationOptions = .{},
};

pub const RuntimeBatchRequest = struct {
    items: []const RuntimeRequest,
};

pub const PlanBatch = struct {
    allocator: std.mem.Allocator,
    request_indices: []usize,
    operation: []const u8,
    execution: ExecutionMode,
    allows_batching: bool,

    pub fn deinit(self: *PlanBatch) void {
        self.allocator.free(self.request_indices);
        self.* = undefined;
    }
};

pub const ExecutionPath = enum {
    shared,
    native,
    fallback,
};

pub const ExecutionPlan = struct {
    allocator: std.mem.Allocator,
    model_id: []const u8,
    request_count: usize,
    execution: ExecutionMode,
    path: ExecutionPath,
    batches: []PlanBatch,

    pub fn deinit(self: *ExecutionPlan) void {
        for (self.batches) |*batch| batch.deinit();
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

pub const RuntimeResult = struct {
    origin: ExecutionOrigin = .shared_adapter,
    note: []const u8 = "",
    output: OutputPayload = .none,
    diagnostics: []const Diagnostic = &.{},

    pub fn deinit(self: *RuntimeResult, allocator: std.mem.Allocator) void {
        self.output.deinit(allocator);
        self.* = undefined;
    }
};

pub fn inputKind(payload: InputPayload) InputKind {
    return switch (payload) {
        .none => .none,
        .text => .text,
        .image_path => .image_path,
        .audio_path => .audio_path,
        .video_path => .video_path,
    };
}
