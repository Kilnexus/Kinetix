const std = @import("std");
const graph = @import("graph");
const blocks = @import("../modules/blocks.zig");
const graph_exec = @import("graph_exec.zig");
const types = @import("engine_vision_base").types;
const weights_mod = @import("weights");

pub const Tensor = types.Tensor;

pub const NodeTrace = struct {
    index: usize,
    path: []u8,
    kind: []u8,
    shape: [4]usize,
    min: f32,
    max: f32,
    mean: f32,
    l2: f32,
    first: f32,

    pub fn deinit(self: *NodeTrace, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.kind);
        self.* = undefined;
    }
};

pub const GraphTrace = struct {
    allocator: std.mem.Allocator,
    nodes: []NodeTrace,

    pub fn deinit(self: *GraphTrace) void {
        for (self.nodes) |*node| node.deinit(self.allocator);
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
};

pub fn traceGraph(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    input: *const Tensor,
) !GraphTrace {
    var outputs = try allocator.alloc(?Tensor, model_graph.execution_nodes.len);
    defer allocator.free(outputs);
    for (outputs) |*item| item.* = null;
    defer {
        for (outputs) |*item| {
            if (item.*) |*tensor| tensor.deinit();
        }
    }

    var traces = try allocator.alloc(NodeTrace, model_graph.execution_nodes.len - 1);
    var trace_count: usize = 0;
    errdefer {
        for (traces[0..trace_count]) |*trace| trace.deinit(allocator);
        allocator.free(traces);
    }

    for (model_graph.execution_nodes, 0..) |*node, node_index| {
        if (std.mem.eql(u8, node.kind, "Detect")) continue;

        const output = if (std.mem.eql(u8, node.kind, "Concat")) blk: {
            var tensor_ptrs = try allocator.alloc(*const Tensor, node.from.len);
            defer allocator.free(tensor_ptrs);

            var channels: usize = 0;
            var height: usize = 0;
            var width: usize = 0;
            for (node.from, 0..) |source, source_index| {
                const tensor = graph_exec.resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
                tensor_ptrs[source_index] = tensor;
                channels += tensor.shape[1];
                if (source_index == 0) {
                    height = tensor.shape[2];
                    width = tensor.shape[3];
                }
            }

            var merged = try Tensor.init(allocator, tensor_ptrs[0].shape[0], channels, height, width);
            errdefer merged.deinit();
            try @import("ops").concatChannels(tensor_ptrs, &merged);
            break :blk merged;
        } else blk: {
            const source = graph_exec.resolveInput(node.from[0], node_index, input, outputs) orelse return error.ModuleNotFound;
            var module_path_buffer: [256]u8 = undefined;
            const module_path = try graph_exec.modulePathForNode(&module_path_buffer, node.path);

            if (std.mem.eql(u8, node.kind, "Upsample")) {
                break :blk try graph_exec.runUpsampleModule(allocator, model_graph, module_path, source);
            }
            break :blk try blocks.runModule(allocator, model_graph, weights_blob, module_path, source);
        };

        outputs[node_index] = output;
        traces[trace_count] = try summarizeTensor(allocator, node, &output);
        trace_count += 1;
    }

    return .{
        .allocator = allocator,
        .nodes = traces[0..trace_count],
    };
}

fn summarizeTensor(allocator: std.mem.Allocator, node: *const graph.ExecutionNode, tensor: *const Tensor) !NodeTrace {
    var min_value = tensor.data[0];
    var max_value = tensor.data[0];
    var sum: f64 = 0.0;
    var sq_sum: f64 = 0.0;
    for (tensor.data) |value| {
        if (value < min_value) min_value = value;
        if (value > max_value) max_value = value;
        sum += value;
        sq_sum += @as(f64, value) * @as(f64, value);
    }

    return .{
        .index = node.index,
        .path = try allocator.dupe(u8, node.path),
        .kind = try allocator.dupe(u8, node.kind),
        .shape = tensor.shape,
        .min = min_value,
        .max = max_value,
        .mean = @floatCast(sum / @as(f64, @floatFromInt(tensor.data.len))),
        .l2 = @floatCast(@sqrt(sq_sum)),
        .first = tensor.data[0],
    };
}
