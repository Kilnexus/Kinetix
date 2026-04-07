const std = @import("std");
const cache = @import("cache.zig");
const graph_types = @import("types.zig");

pub const Graph = graph_types.Graph;
pub const ModuleNode = graph_types.ModuleNode;

pub fn indexGraph(model_graph: *Graph) !void {
    try model_graph.tensor_index.ensureTotalCapacity(model_graph.allocator, @intCast(model_graph.tensors.len));
    for (model_graph.tensors) |*tensor| {
        model_graph.tensor_index.putAssumeCapacity(tensor.name, tensor);
    }

    const module_count = countModules(&model_graph.module_tree);
    try model_graph.module_index.ensureTotalCapacity(model_graph.allocator, @intCast(module_count));
    indexModuleNode(&model_graph.module_index, &model_graph.module_tree);
    cache.cacheConvSpecs(model_graph, &model_graph.module_tree);
    cache.cacheModuleAttrs(&model_graph.module_tree);

    for (model_graph.execution_nodes, 0..) |*node, index| {
        model_graph.execution_modules[index] = executionModuleForPath(model_graph, node.path);
    }
    buildExecutionUseCounts(model_graph);
}

fn countModules(node: *const ModuleNode) usize {
    var total: usize = 1;
    for (node.children) |*child| total += countModules(child);
    return total;
}

fn indexModuleNode(index: *std.StringHashMapUnmanaged(*const ModuleNode), node: *const ModuleNode) void {
    index.putAssumeCapacity(node.path, node);
    for (node.children) |*child| indexModuleNode(index, child);
}

fn executionModuleForPath(model_graph: *const Graph, path: []const u8) ?*const ModuleNode {
    if (!std.mem.startsWith(u8, path, "model.")) return null;

    var buffer: [256]u8 = undefined;
    const module_path = std.fmt.bufPrint(&buffer, "model.model.{s}", .{path["model.".len..]}) catch return null;
    return model_graph.findModule(module_path);
}

fn buildExecutionUseCounts(model_graph: *Graph) void {
    @memset(model_graph.execution_use_counts, 0);
    for (model_graph.execution_nodes, 0..) |node, node_index| {
        for (node.from) |source| {
            if (source == -1) {
                if (node_index > 0) model_graph.execution_use_counts[node_index - 1] += 1;
            } else {
                model_graph.execution_use_counts[@intCast(source)] += 1;
            }
        }
    }
}
