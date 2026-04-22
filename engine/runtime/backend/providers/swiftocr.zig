const std = @import("std");
const backend_mod = @import("../backend.zig");
const handle_mod = @import("../../model/handle.zig");
const swiftocr_native = @import("../../providers/swiftocr_native.zig");
const types = @import("../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .swiftocr_ocr,
    .execute_fn = execute,
};

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    switch (request.input) {
        .image_path => {},
        else => return error.InvalidInputPayload,
    }
    const input_path = request.input.asString() orelse return error.MissingInputPayload;
    const model_path = handle.normalized.artifacts.ocr_model_path orelse return error.MissingOCRModelArtifact;
    const output = try swiftocr_native.execute(allocator, .{
        .operation = request.operation,
        .model_path = model_path,
        .input_path = input_path,
        .execution = request.execution,
    });
    return .{
        .origin = .shared_adapter,
        .note = .ocr_swiftocr_native,
        .output = .{ .json = output },
    };
}
