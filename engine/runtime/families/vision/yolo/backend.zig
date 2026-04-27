const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const types = @import("../../../types.zig");
const vision_shared = @import("../../../shared/vision/runtime.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .yolo_vision,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    summary: ?vision_shared.Summary = null,
    graph: ?vision_shared.Graph = null,
    weights: ?vision_shared.WeightsBlob = null,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        if (self.weights) |*weights| weights.deinit();
        if (self.graph) |*graph| graph.deinit();
        if (self.summary) |summary| allocator.free(summary.model_name);
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
            .provider_key = .yolo_vision,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .summary = null,
        .graph = null,
        .weights = null,
    };
    errdefer allocator.free(state.base.model_dir);

    if (model.artifacts.graph_path) |graph_path| {
        state.summary = vision_shared.loadSummary(allocator, graph_path) catch null;
    }
    errdefer if (state.summary) |summary| allocator.free(summary.model_name);

    if (!builtin.is_test) {
        if (model.artifacts.graph_path) |graph_path| {
            state.graph = vision_shared.loadGraph(allocator, graph_path) catch null;
        }
        errdefer if (state.graph) |*graph| graph.deinit();

        if (model.artifacts.binary_weights_path) |weights_path| {
            state.weights = vision_shared.loadWeights(allocator, weights_path) catch null;
        }
        errdefer if (state.weights) |*weights| weights.deinit();
    }

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
    switch (request.input) {
        .none, .image_path => {},
        else => return error.InvalidInputPayload,
    }

    const graph_path = handle.normalized.artifacts.graph_path orelse return error.MissingGraphArtifact;
    const weights_path = handle.normalized.artifacts.binary_weights_path orelse return error.MissingBinaryWeightsArtifact;
    const maybe_state = stateFromHandle(handle);
    const loaded_summary = if (maybe_state == null or maybe_state.?.summary == null)
        try vision_shared.loadSummary(allocator, graph_path)
    else
        null;
    defer if (loaded_summary) |summary| allocator.free(summary.model_name);
    const summary = if (maybe_state) |state|
        state.summary orelse loaded_summary.?
    else
        loaded_summary.?;

    var detection_output = detect: {
        if (maybe_state) |state| {
            if (state.graph) |*model_graph| {
                if (state.weights) |*weights_blob| {
                    break :detect try vision_shared.maybeRunDetectWithLoaded(
                        allocator,
                        model_graph,
                        weights_blob,
                        request.operation,
                        request.execution,
                        request.input.asString(),
                    );
                }
            }
        }
        break :detect try vision_shared.maybeRunDetect(
            allocator,
            graph_path,
            weights_path,
            request.operation,
            request.execution,
            request.input.asString(),
        );
    };
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
        .origin = .runtime_backend,
        .note = if (detection_output != null) .vision_shared_detect else .vision_graph_ready,
        .output = .{ .json = output },
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
