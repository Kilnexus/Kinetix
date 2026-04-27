const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const types = @import("../../../types.zig");
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
