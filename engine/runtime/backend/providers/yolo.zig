const std = @import("std");
const backend_mod = @import("../backend.zig");
const handle_mod = @import("../../model/handle.zig");
const types = @import("../../types.zig");
const vision_shared = @import("../../providers/vision_shared.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .yolo_vision,
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
        .none, .image_path => {},
        else => return error.InvalidInputPayload,
    }

    const graph_path = handle.normalized.artifacts.graph_path orelse return error.MissingGraphArtifact;
    const weights_path = handle.normalized.artifacts.binary_weights_path orelse return error.MissingBinaryWeightsArtifact;
    const summary = try vision_shared.loadSummary(allocator, graph_path);
    defer allocator.free(summary.model_name);

    var detection_output = try vision_shared.maybeRunDetect(
        allocator,
        graph_path,
        weights_path,
        request.operation,
        request.execution,
        request.input.asString(),
    );
    defer if (detection_output) |*output| output.deinit(allocator);

    const output = try vision_shared.buildOutputJson(allocator, .{
        .operation = request.operation,
        .model_name = summary.model_name,
        .model_family = handle.normalized.descriptor.family,
        .input_path = request.input.asString(),
        .execution_nodes = summary.execution_nodes,
        .tensor_count = summary.tensor_count,
        .class_count = summary.class_count,
    }, detection_output);
    return .{
        .origin = .shared_adapter,
        .note = if (detection_output != null) .vision_shared_detect else .vision_graph_ready,
        .output = .{ .json = output },
    };
}
