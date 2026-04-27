const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const types = @import("../../../types.zig");
const inference = @import("inference/index.zig");
const resolver = @import("resolver.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .paddleocr_ocr,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    layout: resolver.Layout,
    onnx_metadata: inference.planning.onnx_metadata.Summary = .{},

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.base.model_dir);
        allocator.destroy(self);
    }
};

fn open(
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) !?*anyopaque {
    const state = try allocator.create(State);
    errdefer allocator.destroy(state);
    state.* = .{
        .base = .{
            .provider_key = .paddleocr_ocr,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .layout = resolver.inspectLayout(model.artifacts.model_dir),
        .onnx_metadata = inference.planning.onnx_metadata.inspect(allocator, model.artifacts.model_dir) catch inference.planning.onnx_metadata.Summary{},
    };
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
    if (request.resolvedOperationId() != .ocr) return error.OperationNotSupported;
    switch (request.input) {
        .image_path, .document_path => {},
        else => return error.InvalidInputPayload,
    }

    const state = stateFromHandle(handle);
    const layout = if (state) |loaded| loaded.layout else resolver.inspectLayout(handle.normalized.artifacts.model_dir);
    const onnx_metadata = if (state) |loaded|
        loaded.onnx_metadata
    else
        inference.planning.onnx_metadata.inspect(allocator, handle.normalized.artifacts.model_dir) catch inference.planning.onnx_metadata.Summary{};

    const Receipt = struct {
        status: []const u8,
        provider_key: []const u8,
        model_family: []const u8,
        model_id: []const u8,
        operation: []const u8,
        input: ?[]const u8,
        backend: []const u8,
        runtime_contract: []const u8,
        has_paddle_inference_model: bool,
        has_paddle_inference_params: bool,
        has_inference_yml: bool,
        has_model_yml: bool,
        has_rec_dict: bool,
        has_onnx_model: bool,
        det_model_count: usize,
        rec_model_count: usize,
        cls_model_count: usize,
        onnx_metadata_loaded_graph_count: usize,
        onnx_metadata_det_graph_count: usize,
        onnx_metadata_rec_graph_count: usize,
        onnx_metadata_cls_graph_count: usize,
        onnx_metadata_total_node_count: usize,
        onnx_metadata_total_initializer_count: usize,
        onnx_metadata_external_initializer_count: usize,
        onnx_metadata_supported_node_count: usize,
        onnx_metadata_unsupported_node_count: usize,
        onnx_metadata_unsupported_ops: []const inference.planning.onnx_metadata.UnsupportedOpEntry,
        message: []const u8,
    };

    const receipt = Receipt{
        .status = "paddleocr_runtime_ready",
        .provider_key = handle.normalized.provider_key.name(),
        .model_family = handle.normalized.descriptor.family,
        .model_id = handle.normalized.descriptor.id,
        .operation = request.operation,
        .input = request.input.asString(),
        .backend = "kinetix_native",
        .runtime_contract = "pp_ocr_pipeline",
        .has_paddle_inference_model = layout.has_paddle_inference_model,
        .has_paddle_inference_params = layout.has_paddle_inference_params,
        .has_inference_yml = layout.has_inference_yml,
        .has_model_yml = layout.has_model_yml,
        .has_rec_dict = layout.has_rec_dict,
        .has_onnx_model = layout.has_onnx_model,
        .det_model_count = layout.det_model_count,
        .rec_model_count = layout.rec_model_count,
        .cls_model_count = layout.cls_model_count,
        .onnx_metadata_loaded_graph_count = onnx_metadata.loaded_graph_count,
        .onnx_metadata_det_graph_count = onnx_metadata.det_graph_count,
        .onnx_metadata_rec_graph_count = onnx_metadata.rec_graph_count,
        .onnx_metadata_cls_graph_count = onnx_metadata.cls_graph_count,
        .onnx_metadata_total_node_count = onnx_metadata.total_node_count,
        .onnx_metadata_total_initializer_count = onnx_metadata.total_initializer_count,
        .onnx_metadata_external_initializer_count = onnx_metadata.external_initializer_count,
        .onnx_metadata_supported_node_count = onnx_metadata.supported_node_count,
        .onnx_metadata_unsupported_node_count = onnx_metadata.unsupported_node_count,
        .onnx_metadata_unsupported_ops = onnx_metadata.unsupported_ops[0..onnx_metadata.unsupported_op_entry_count],
        .message = "PaddleOCR is routed through the unified runtime. Native zero-dependency PP-OCRv5 detection, recognition, classification, and postprocess execution will be enabled by expanding shared graph ops and Paddle/ONNX model loading.",
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);

    return .{
        .origin = .runtime_backend,
        .note = .ocr_model_ready,
        .output = .{ .json = try allocator.dupe(u8, out.written()) },
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
