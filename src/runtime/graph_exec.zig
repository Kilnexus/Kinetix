const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const detect = @import("detect.zig");
const execute = @import("execute.zig");
const types = @import("types.zig");
const weights_mod = @import("weights");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;
pub const DetectOptions = detect.DetectOptions;
pub const DetectOutput = detect.DetectOutput;

pub fn runUpsampleModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "Upsample")) return error.InvalidModuleKind;

    const scale_value = (module.getAttr("scale_factor") orelse return error.MissingAttribute).asFloat() orelse return error.InvalidAttributeType;
    const scale: usize = @intFromFloat(scale_value);

    var output = try Tensor.init(
        allocator,
        input.shape[0],
        input.shape[1],
        input.shape[2] * scale,
        input.shape[3] * scale,
    );
    errdefer output.deinit();

    try ops.upsampleNearest(input, &output, scale, scale);
    return output;
}

pub fn runGraph(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    input: *const Tensor,
    detect_options: DetectOptions,
) !DetectOutput {
    var outputs = try allocator.alloc(?Tensor, model_graph.execution_nodes.len);
    defer allocator.free(outputs);
    for (outputs) |*item| item.* = null;
    defer {
        for (outputs) |*item| {
            if (item.*) |*tensor| tensor.deinit();
        }
    }

    var detect_output: ?DetectOutput = null;

    for (model_graph.execution_nodes, 0..) |*node, node_index| {
        if (std.mem.eql(u8, node.kind, "Detect")) {
            var feature_ptrs = try allocator.alloc(*const Tensor, node.from.len);
            defer allocator.free(feature_ptrs);
            for (node.from, 0..) |source, source_index| {
                feature_ptrs[source_index] = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
            }
            var module_path_buffer: [256]u8 = undefined;
            detect_output = try detect.runDetect(
                allocator,
                model_graph,
                weights_blob,
                try modulePathForNode(&module_path_buffer, node.path),
                feature_ptrs,
                detect_options,
            );
            continue;
        }

        if (std.mem.eql(u8, node.kind, "Concat")) {
            var tensor_ptrs = try allocator.alloc(*const Tensor, node.from.len);
            defer allocator.free(tensor_ptrs);

            var channels: usize = 0;
            var height: usize = 0;
            var width: usize = 0;
            for (node.from, 0..) |source, source_index| {
                const tensor = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
                tensor_ptrs[source_index] = tensor;
                channels += tensor.shape[1];
                if (source_index == 0) {
                    height = tensor.shape[2];
                    width = tensor.shape[3];
                }
            }

            var merged = try Tensor.init(allocator, tensor_ptrs[0].shape[0], channels, height, width);
            errdefer merged.deinit();
            try ops.concatChannels(tensor_ptrs, &merged);
            outputs[node_index] = merged;
            continue;
        }

        const source = resolveInput(node.from[0], node_index, input, outputs) orelse return error.ModuleNotFound;
        var module_path_buffer: [256]u8 = undefined;
        const module_path = try modulePathForNode(&module_path_buffer, node.path);

        const output = if (std.mem.eql(u8, node.kind, "Upsample"))
            try runUpsampleModule(allocator, model_graph, module_path, source)
        else
            try execute.runModule(allocator, model_graph, weights_blob, module_path, source);
        outputs[node_index] = output;
    }

    return detect_output orelse error.ModuleNotFound;
}

fn resolveInput(
    from: i64,
    node_index: usize,
    input: *const Tensor,
    outputs: []?Tensor,
) ?*const Tensor {
    if (from == -1) {
        if (node_index == 0) return input;
        if (outputs[node_index - 1]) |*tensor| return tensor;
        return null;
    }

    const index: usize = @intCast(from);
    if (index >= outputs.len) return null;
    if (outputs[index]) |*tensor| return tensor;
    return null;
}

fn modulePathForNode(buffer: []u8, path: []const u8) RuntimeError![]const u8 {
    if (!std.mem.startsWith(u8, path, "model.")) return error.InvalidAttributeType;
    return std.fmt.bufPrint(buffer, "model.model.{s}", .{path["model.".len..]}) catch return error.BufferTooSmall;
}
