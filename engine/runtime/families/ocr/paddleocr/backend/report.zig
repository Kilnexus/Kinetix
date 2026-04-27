const std = @import("std");
const types = @import("../../../../types.zig");
const inference = @import("../inference/index.zig");
const postprocess = @import("../postprocess/index.zig");
const resolver = @import("../resolver.zig");

pub const Input = struct {
    provider_key: []const u8,
    model_family: []const u8,
    model_id: []const u8,
    operation: []const u8,
    input: ?[]const u8,
    layout: resolver.Layout,
    onnx_metadata: inference.planning.onnx_metadata.Summary,
    image_graph_model_path: ?[]const u8,
    image_graph_stage: inference.runtime.StageKind,
    image_graph_result: ?*const inference.runtime.ImageModelResult,
    image_graph_error: ?[]const u8,
    detection_box_count: usize,
    recognized_text: ?*const postprocess.ctc.DecodedText,
    dictionary_path: ?[]const u8,
    postprocess_error: ?[]const u8,
    pipeline_result: ?*const inference.runtime.PipelineResult,
    pipeline_error: ?[]const u8,
};

pub fn runtimeResult(allocator: std.mem.Allocator, input: Input) !types.RuntimeResult {
    const lines = try ocrLinesFromPipeline(allocator, input.pipeline_result);
    defer allocator.free(lines);
    const receipt = receiptFromInput(input, lines);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);

    return .{
        .origin = .runtime_backend,
        .note = .ocr_model_ready,
        .output = .{ .json = try allocator.dupe(u8, out.written()) },
    };
}

fn receiptFromInput(input: Input, lines: []const OcrLine) Receipt {
    const image_result = input.image_graph_result;
    const pipeline = input.pipeline_result;
    const recognized = input.recognized_text;
    const ocr_text = if (pipeline) |result| result.text else "";

    return .{
        .status = if (ocr_text.len != 0) "ocr_infer_completed" else "paddleocr_runtime_ready",
        .provider_key = input.provider_key,
        .model_family = input.model_family,
        .model_id = input.model_id,
        .operation = input.operation,
        .input = input.input,
        .backend = "kinetix_native",
        .runtime_contract = "pp_ocr_pipeline",
        .has_paddle_inference_model = input.layout.has_paddle_inference_model,
        .has_paddle_inference_params = input.layout.has_paddle_inference_params,
        .has_inference_yml = input.layout.has_inference_yml,
        .has_model_yml = input.layout.has_model_yml,
        .has_rec_dict = input.layout.has_rec_dict,
        .has_onnx_model = input.layout.has_onnx_model,
        .det_model_count = input.layout.det_model_count,
        .rec_model_count = input.layout.rec_model_count,
        .cls_model_count = input.layout.cls_model_count,
        .onnx_metadata_loaded_graph_count = input.onnx_metadata.loaded_graph_count,
        .onnx_metadata_det_graph_count = input.onnx_metadata.det_graph_count,
        .onnx_metadata_rec_graph_count = input.onnx_metadata.rec_graph_count,
        .onnx_metadata_cls_graph_count = input.onnx_metadata.cls_graph_count,
        .onnx_metadata_total_node_count = input.onnx_metadata.total_node_count,
        .onnx_metadata_total_initializer_count = input.onnx_metadata.total_initializer_count,
        .onnx_metadata_external_initializer_count = input.onnx_metadata.external_initializer_count,
        .onnx_metadata_supported_node_count = input.onnx_metadata.supported_node_count,
        .onnx_metadata_unsupported_node_count = input.onnx_metadata.unsupported_node_count,
        .onnx_metadata_unsupported_ops = input.onnx_metadata.unsupported_ops[0..input.onnx_metadata.unsupported_op_entry_count],
        .ctc_postprocess_ready = @hasDecl(postprocess, "ctc"),
        .db_postprocess_ready = @hasDecl(postprocess, "db"),
        .image_graph_execute_attempted = input.input != null and input.image_graph_model_path != null,
        .image_graph_executed = image_result != null,
        .image_graph_model_path = input.image_graph_model_path,
        .image_graph_stage = input.image_graph_stage.name(),
        .image_graph_input_name = if (image_result) |result| result.input_name else null,
        .image_graph_input_shape = if (image_result) |result| result.input_shape else null,
        .image_graph_output_count = if (image_result) |result| result.outputs.outputs.len else 0,
        .image_graph_first_output_name = if (image_result) |result| if (result.outputs.outputs.len != 0) result.outputs.outputs[0].name else null else null,
        .image_graph_first_output_shape = if (image_result) |result| if (result.outputs.outputs.len != 0) result.outputs.outputs[0].tensor.shape else null else null,
        .image_graph_error = input.image_graph_error,
        .detection_box_count = input.detection_box_count,
        .recognition_text = if (recognized) |text| text.text else null,
        .recognition_token_count = if (recognized) |text| text.token_ids.len else 0,
        .dictionary_path = input.dictionary_path,
        .postprocess_error = input.postprocess_error,
        .pipeline_attempted = input.input != null,
        .pipeline_det_model_path = if (pipeline) |result| result.det_model_path else null,
        .pipeline_rec_model_path = if (pipeline) |result| result.rec_model_path else null,
        .pipeline_cls_model_path = if (pipeline) |result| result.cls_model_path else null,
        .pipeline_dictionary_path = if (pipeline) |result| result.dictionary_path else null,
        .pipeline_det_executed = if (pipeline) |result| result.det_executed else false,
        .pipeline_detection_box_count = if (pipeline) |result| result.detection_boxes.len else 0,
        .pipeline_detection_map_width = if (pipeline) |result| result.detection_map_width else 0,
        .pipeline_detection_map_height = if (pipeline) |result| result.detection_map_height else 0,
        .pipeline_cls_attempted_count = if (pipeline) |result| result.cls_attempted_count else 0,
        .pipeline_cls_rotate_180_count = if (pipeline) |result| result.cls_rotate_180_count else 0,
        .pipeline_rec_attempted_count = if (pipeline) |result| result.rec_attempted_count else 0,
        .pipeline_rec_decoded_count = if (pipeline) |result| result.rec_decoded_count else 0,
        .pipeline_text = if (pipeline) |result| if (result.text.len != 0) result.text else null else null,
        .pipeline_error = if (pipeline) |result| result.error_message orelse input.pipeline_error else input.pipeline_error,
        .ocr_result = .{
            .schema_version = "kinetix.ocr.v1",
            .text = ocr_text,
            .line_count = lines.len,
            .lines = lines,
        },
        .message = "PaddleOCR is routed through the unified runtime. Native zero-dependency PP-OCR graph execution loads ONNX initializers and runs staged det/cls/rec image graphs.",
    };
}

fn ocrLinesFromPipeline(allocator: std.mem.Allocator, pipeline: ?*const inference.runtime.PipelineResult) ![]OcrLine {
    const result = pipeline orelse return try allocator.alloc(OcrLine, 0);
    const lines = try allocator.alloc(OcrLine, result.lines.len);
    errdefer allocator.free(lines);
    for (result.lines, lines, 0..) |line, *slot, index| {
        slot.* = .{
            .index = index,
            .text = line.text,
            .token_count = line.token_count,
            .box = if (line.box) |box| ocrBox(box) else null,
        };
    }
    return lines;
}

fn ocrBox(box: postprocess.db.Box) OcrBox {
    return .{
        .x_min = box.x_min,
        .y_min = box.y_min,
        .x_max = box.x_max,
        .y_max = box.y_max,
        .score = box.score,
        .points = .{
            .{ .x = box.points[0].x, .y = box.points[0].y },
            .{ .x = box.points[1].x, .y = box.points[1].y },
            .{ .x = box.points[2].x, .y = box.points[2].y },
            .{ .x = box.points[3].x, .y = box.points[3].y },
        },
    };
}

const OcrOutput = struct {
    schema_version: []const u8,
    text: []const u8,
    line_count: usize,
    lines: []const OcrLine,
};

const OcrLine = struct {
    index: usize,
    text: []const u8,
    token_count: usize,
    box: ?OcrBox,
};

const OcrBox = struct {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    score: f32,
    points: [4]OcrPoint,
};

const OcrPoint = struct {
    x: f32,
    y: f32,
};

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
    pipeline_attempted: bool,
    pipeline_det_model_path: ?[]const u8,
    pipeline_rec_model_path: ?[]const u8,
    pipeline_cls_model_path: ?[]const u8,
    pipeline_dictionary_path: ?[]const u8,
    pipeline_det_executed: bool,
    pipeline_detection_box_count: usize,
    pipeline_detection_map_width: usize,
    pipeline_detection_map_height: usize,
    pipeline_cls_attempted_count: usize,
    pipeline_cls_rotate_180_count: usize,
    pipeline_rec_attempted_count: usize,
    pipeline_rec_decoded_count: usize,
    pipeline_text: ?[]const u8,
    pipeline_error: ?[]const u8,
    ocr_result: OcrOutput,
    message: []const u8,
};
