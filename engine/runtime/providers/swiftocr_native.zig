const std = @import("std");
const ocr_pipeline = @import("../ocr_pipeline.zig");
const task = @import("../../core/task.zig");

pub const InferResult = ocr_pipeline.InferResult;

pub const Context = struct {
    operation: []const u8,
    model_path: []const u8,
    input_path: []const u8,
    execution: task.ExecutionMode,
};

pub fn execute(allocator: std.mem.Allocator, context: Context) ![]u8 {
    const infer_output = try maybeRunInfer(allocator, context);
    return try buildOutputJson(allocator, context, infer_output);
}

fn maybeRunInfer(allocator: std.mem.Allocator, context: Context) !?InferResult {
    if (!std.mem.eql(u8, context.operation, "infer-ocr")) return null;
    if (context.execution != .sync) return null;

    var pipeline = ocr_pipeline.OCRPipeline.init(allocator);
    defer pipeline.deinit();
    return try pipeline.infer(.{
        .model_path = context.model_path,
        .image_path = context.input_path,
    });
}

fn buildOutputJson(
    allocator: std.mem.Allocator,
    context: Context,
    infer_output: ?InferResult,
) ![]u8 {
    const OCRReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: []const u8,
        loaded_tensors: ?usize,
        image_width: ?usize,
        image_height: ?usize,
    };

    const receipt = OCRReceipt{
        .status = if (infer_output != null) "ocr_infer_completed" else "ocr_model_ready",
        .operation = context.operation,
        .model_family = "swiftocr",
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

test "swiftocr native execute emits infer receipt" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var model_file = try tmp.dir.createFile("demo.swm", .{});
    defer model_file.close();
    var model_writer_impl = model_file.writer(&.{});
    const model_writer = &model_writer_impl.interface;
    try model_writer.writeAll(&[_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 });
    try model_writer.writeInt(u32, 0, .little);
    try model_writer.flush();

    var image_file = try tmp.dir.createFile("demo.ppm", .{});
    defer image_file.close();
    var image_writer_impl = image_file.writer(&.{});
    const image_writer = &image_writer_impl.interface;
    try image_writer.writeAll("P6\n1 1\n255\n");
    try image_writer.writeAll(&[_]u8{ 1, 2, 3 });
    try image_writer.flush();

    const model_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.swm");
    defer std.testing.allocator.free(model_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.ppm");
    defer std.testing.allocator.free(image_path);

    const payload = try execute(std.testing.allocator, .{
        .operation = "infer-ocr",
        .model_path = model_path,
        .input_path = image_path,
        .execution = .sync,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"ocr_infer_completed\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"model_family\":\"swiftocr\"") != null);
}
