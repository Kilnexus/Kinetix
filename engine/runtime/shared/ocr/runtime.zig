const std = @import("std");
const ocr_pipeline = @import("../../ocr_pipeline.zig");
const task = @import("../../../core/task.zig");

pub const InferResult = ocr_pipeline.InferResult;

pub const ReceiptContext = struct {
    operation: []const u8,
    model_family: []const u8,
    model_path: []const u8,
    input_path: ?[]const u8,
};

pub fn maybeRunInfer(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    operation: []const u8,
    execution: task.ExecutionMode,
    input_path: ?[]const u8,
) !?InferResult {
    if (!std.mem.eql(u8, operation, "infer-ocr")) return null;
    if (execution != .sync) return null;

    const image_path = input_path orelse return null;

    var pipeline = ocr_pipeline.OCRPipeline.init(allocator);
    defer pipeline.deinit();
    return try pipeline.infer(.{
        .model_path = model_path,
        .image_path = image_path,
    });
}

pub fn buildOutputJson(
    allocator: std.mem.Allocator,
    context: ReceiptContext,
    infer_output: ?InferResult,
) ![]u8 {
    const OCRReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: ?[]const u8,
        loaded_tensors: ?usize,
        image_width: ?usize,
        image_height: ?usize,
    };

    const receipt = OCRReceipt{
        .status = if (infer_output != null) "ocr_infer_completed" else "ocr_model_ready",
        .operation = context.operation,
        .model_family = context.model_family,
        .model_path = context.model_path,
        .input_path = context.input_path,
        .loaded_tensors = if (infer_output) |output| output.loaded_tensors else null,
        .image_width = if (infer_output) |output| output.image_width else null,
        .image_height = if (infer_output) |output| output.image_height else null,
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}
