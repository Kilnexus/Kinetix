const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const types = @import("../../../types.zig");
const inference = @import("inference/index.zig");
const postprocess = @import("postprocess/index.zig");
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
    const image_path = switch (request.input) {
        .image_path => request.input.asString(),
        else => null,
    };
    var image_graph_model_path: ?[]u8 = null;
    defer if (image_graph_model_path) |path| allocator.free(path);
    var image_graph_result: ?inference.runtime.ImageModelResult = null;
    defer if (image_graph_result) |*result| result.deinit();
    var image_graph_error: ?[]const u8 = null;
    var postprocess_error: ?[]const u8 = null;
    var detected_boxes: ?[]postprocess.db.Box = null;
    defer if (detected_boxes) |boxes| allocator.free(boxes);
    var dictionary_path: ?[]u8 = null;
    defer if (dictionary_path) |path| allocator.free(path);
    var dictionary: ?postprocess.dictionary.Dictionary = null;
    defer if (dictionary) |*loaded| loaded.deinit();
    var recognized_text: ?postprocess.ctc.DecodedText = null;
    defer if (recognized_text) |*text| text.deinit();
    if (image_path) |path| {
        image_graph_model_path = inference.runtime.findFirstOnnxModelFile(allocator, handle.normalized.artifacts.model_dir) catch |err| blk: {
            image_graph_error = @errorName(err);
            break :blk null;
        };
        if (image_graph_model_path) |model_path| {
            image_graph_result = inference.runtime.executeImageModelFile(allocator, model_path, path, .{}) catch |err| blk: {
                image_graph_error = @errorName(err);
                break :blk null;
            };
        }
    }
    const image_graph_stage = if (image_graph_model_path) |model_path| inference.runtime.stageFromPath(model_path) else .unknown;
    if (image_graph_result) |*result| {
        switch (image_graph_stage) {
            .det => {
                detected_boxes = inference.runtime.boxesFromDetectionResult(allocator, &result.outputs, .{}) catch |err| blk: {
                    postprocess_error = @errorName(err);
                    break :blk null;
                };
            },
            .rec => {
                dictionary_path = inference.runtime.findFirstDictionaryFile(allocator, handle.normalized.artifacts.model_dir) catch |err| blk: {
                    postprocess_error = @errorName(err);
                    break :blk null;
                };
                if (dictionary_path) |path| {
                    dictionary = postprocess.dictionary.loadFromFile(allocator, path) catch |err| blk: {
                        postprocess_error = @errorName(err);
                        break :blk null;
                    };
                }
                if (dictionary) |loaded| {
                    recognized_text = inference.runtime.decodeRecognitionResult(allocator, &result.outputs, loaded.tokens) catch |err| blk: {
                        postprocess_error = @errorName(err);
                        break :blk null;
                    };
                }
            },
            else => {},
        }
    }

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
        ctc_postprocess_ready: bool,
        db_postprocess_ready: bool,
        image_graph_execute_attempted: bool,
        image_graph_executed: bool,
        image_graph_model_path: ?[]const u8,
        image_graph_stage: []const u8,
        image_graph_input_name: ?[]const u8,
        image_graph_input_shape: ?[]const usize,
        image_graph_output_count: usize,
        image_graph_first_output_name: ?[]const u8,
        image_graph_first_output_shape: ?[]const usize,
        image_graph_error: ?[]const u8,
        detection_box_count: usize,
        recognition_text: ?[]const u8,
        recognition_token_count: usize,
        dictionary_path: ?[]const u8,
        postprocess_error: ?[]const u8,
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
        .ctc_postprocess_ready = @hasDecl(postprocess, "ctc"),
        .db_postprocess_ready = @hasDecl(postprocess, "db"),
        .image_graph_execute_attempted = image_path != null and image_graph_model_path != null,
        .image_graph_executed = image_graph_result != null,
        .image_graph_model_path = image_graph_model_path,
        .image_graph_stage = image_graph_stage.name(),
        .image_graph_input_name = if (image_graph_result) |result| result.input_name else null,
        .image_graph_input_shape = if (image_graph_result) |result| result.input_shape else null,
        .image_graph_output_count = if (image_graph_result) |result| result.outputs.outputs.len else 0,
        .image_graph_first_output_name = if (image_graph_result) |result| if (result.outputs.outputs.len != 0) result.outputs.outputs[0].name else null else null,
        .image_graph_first_output_shape = if (image_graph_result) |result| if (result.outputs.outputs.len != 0) result.outputs.outputs[0].tensor.shape else null else null,
        .image_graph_error = image_graph_error,
        .detection_box_count = if (detected_boxes) |boxes| boxes.len else 0,
        .recognition_text = if (recognized_text) |text| text.text else null,
        .recognition_token_count = if (recognized_text) |text| text.token_ids.len else 0,
        .dictionary_path = dictionary_path,
        .postprocess_error = postprocess_error,
        .message = "PaddleOCR is routed through the unified runtime. Native zero-dependency PP-OCRv5 graph execution now loads ONNX initializers and runs static NCHW image graphs; full det-to-rec multi-stage OCR quality depends on remaining operator coverage and model-specific dynamic-shape handling.",
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
