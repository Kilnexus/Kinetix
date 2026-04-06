const std = @import("std");
const graph_types = @import("types.zig");

pub const ComponentNode = graph_types.ComponentNode;
pub const PlanGraph = graph_types.PlanGraph;

pub fn indexGraph(graph: *PlanGraph) !void {
    try graph.tensor_index.ensureTotalCapacity(graph.allocator, @intCast(graph.tensors.len));
    for (graph.tensors) |*tensor| {
        graph.tensor_index.putAssumeCapacity(tensor.name, tensor);
    }

    const component_count = countComponents(&graph.component_tree);
    try graph.component_index.ensureTotalCapacity(graph.allocator, @intCast(component_count));
    indexComponentNode(&graph.component_index, &graph.component_tree);

    for (graph.execution_nodes, 0..) |*node, index| {
        graph.execution_components[index] = executionComponentForPath(graph, node.path);
    }
    buildExecutionUseCounts(graph);
}

fn countComponents(node: *const ComponentNode) usize {
    var total: usize = 1;
    for (node.children) |*child| total += countComponents(child);
    return total;
}

fn indexComponentNode(index: *std.StringHashMapUnmanaged(*const ComponentNode), node: *const ComponentNode) void {
    index.putAssumeCapacity(node.path, node);
    for (node.children) |*child| indexComponentNode(index, child);
}

fn executionComponentForPath(graph: *const PlanGraph, path: []const u8) ?*const ComponentNode {
    if (graph.findComponent(path)) |component| return component;

    const root_path = graph.component_tree.path;
    if (std.mem.startsWith(u8, path, "model.") and std.mem.eql(u8, root_path, "model")) {
        var buffer: [256]u8 = undefined;
        const rebased = std.fmt.bufPrint(&buffer, "model.model.{s}", .{path["model.".len..]}) catch return null;
        return graph.findComponent(rebased);
    }

    return null;
}

fn buildExecutionUseCounts(graph: *PlanGraph) void {
    @memset(graph.execution_use_counts, 0);
    for (graph.execution_nodes, 0..) |node, node_index| {
        for (node.from) |source| {
            if (source == -1) {
                if (node_index > 0) graph.execution_use_counts[node_index - 1] += 1;
            } else {
                graph.execution_use_counts[@intCast(source)] += 1;
            }
        }
    }
}
