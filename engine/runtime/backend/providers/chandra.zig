const std = @import("std");
const backend_mod = @import("../backend.zig");
const chandra_native = @import("../../providers/chandra_native.zig");
const handle_mod = @import("../../model/handle.zig");
const types = @import("../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .chandra_ocr,
    .open_fn = backend_mod.openBasicState,
    .deinit_fn = backend_mod.deinitBasicState,
    .execute_fn = execute,
};

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    switch (request.input) {
        .image_path, .document_path => {},
        else => return error.InvalidInputPayload,
    }
    const input_path = request.input.asString() orelse return error.MissingInputPayload;
    const context = chandra_native.Context{
        .operation = request.operation,
        .model_path = handle.normalized.artifacts.model_dir,
        .input_path = input_path,
        .execution = request.execution,
        .max_output_tokens = request.generation.max_tokens,
    };
    const output = try chandra_native.execute(allocator, context);
    return .{
        .origin = .shared_adapter,
        .note = .ocr_chandra_native,
        .output = .{ .json = output },
    };
}
