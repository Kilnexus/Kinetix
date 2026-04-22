const std = @import("std");
const backend_mod = @import("../../backend/backend.zig");
const handle_mod = @import("../../model/handle.zig");
const types = @import("../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .generic,
    .open_fn = backend_mod.openBasicState,
    .deinit_fn = backend_mod.deinitBasicState,
    .execute_fn = execute,
};

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    const Receipt = struct {
        status: []const u8,
        provider_key: []const u8,
        modality: []const u8,
        model_family: []const u8,
        model_id: []const u8,
        operation: []const u8,
        input: ?[]const u8,
        message: []const u8,
    };

    const receipt = Receipt{
        .status = "runtime_backend_ready",
        .provider_key = handle.normalized.provider_key.name(),
        .modality = @tagName(handle.normalized.descriptor.modality),
        .model_family = handle.normalized.descriptor.family,
        .model_id = handle.normalized.descriptor.id,
        .operation = request.operation,
        .input = request.input.asString(),
        .message = "This model is now routed through the unified runtime backend, but no specialized execution backend has been bound for this generic family yet.",
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);

    return .{
        .origin = .shared_adapter,
        .note = .validated_only,
        .output = .{ .json = try allocator.dupe(u8, out.written()) },
    };
}
