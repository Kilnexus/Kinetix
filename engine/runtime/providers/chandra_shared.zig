const std = @import("std");
const task = @import("../../core/task.zig");

pub const ReceiptContext = struct {
    operation: []const u8,
    model_family: []const u8,
    model_path: []const u8,
    input_path: ?[]const u8,
};

pub fn buildOutputJson(
    allocator: std.mem.Allocator,
    context: ReceiptContext,
) ![]u8 {
    const Receipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: ?[]const u8,
        backend: []const u8,
        requested_output: []const u8,
        execution_mode: []const u8,
    };

    const receipt = Receipt{
        .status = "chandra_runtime_pending",
        .operation = context.operation,
        .model_family = context.model_family,
        .model_path = context.model_path,
        .input_path = context.input_path,
        .backend = "external_runtime_required",
        .requested_output = requestedOutput(context.operation),
        .execution_mode = "sync",
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
    return "json";
}
