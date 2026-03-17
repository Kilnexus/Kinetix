const std = @import("std");

pub const TensorMeta = struct {
    name: []u8,
    rank: usize,
    shape: [4]usize,
    offset: usize,
    nbytes: usize,

    pub fn floatLen(self: *const TensorMeta) usize {
        return self.nbytes / @sizeOf(f32);
    }
};

pub const ExecutionNode = struct {
    index: usize,
    path: []u8,
    kind: []u8,
    from: []i64,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    format_version: i64,
    model_name: []u8,
    class_count: i64,
    tensors: []TensorMeta,
    execution_nodes: []ExecutionNode,

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.model_name);
        for (self.tensors) |tensor| self.allocator.free(tensor.name);
        self.allocator.free(self.tensors);
        for (self.execution_nodes) |node| {
            self.allocator.free(node.path);
            self.allocator.free(node.kind);
            self.allocator.free(node.from);
        }
        self.allocator.free(self.execution_nodes);
        self.* = undefined;
    }
};

pub const Summary = struct {
    format_version: i64,
    model_name: []const u8,
    tensor_count: usize,
    execution_nodes: usize,
    class_count: i64,
};

pub fn load(allocator: std.mem.Allocator, graph_path: []const u8) !Graph {
    const cwd = std.fs.cwd();
    const contents = try cwd.readFileAlloc(allocator, graph_path, 64 * 1024 * 1024);
    defer allocator.free(contents);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const metadata = root.get("metadata").?.object;
    const tensors_json = root.get("tensors").?.array;
    const execution_plan_json = root.get("execution_plan").?.array;

    var tensors = try allocator.alloc(TensorMeta, tensors_json.items.len);
    errdefer allocator.free(tensors);
    for (tensors_json.items, 0..) |tensor_value, index| {
        const tensor_object = tensor_value.object;
        const shape_json = tensor_object.get("shape").?.array;
        var shape = [_]usize{ 1, 1, 1, 1 };
        for (shape_json.items, 0..) |dim_value, dim_index| {
            if (dim_index >= shape.len) break;
            shape[dim_index] = @intCast(dim_value.integer);
        }

        tensors[index] = .{
            .name = try allocator.dupe(u8, tensor_object.get("name").?.string),
            .rank = shape_json.items.len,
            .shape = shape,
            .offset = @intCast(tensor_object.get("offset").?.integer),
            .nbytes = @intCast(tensor_object.get("nbytes").?.integer),
        };
    }

    var execution_nodes = try allocator.alloc(ExecutionNode, execution_plan_json.items.len);
    errdefer {
        for (execution_nodes[0..execution_plan_json.items.len]) |node| {
            allocator.free(node.path);
            allocator.free(node.kind);
            allocator.free(node.from);
        }
        allocator.free(execution_nodes);
    }
    for (execution_plan_json.items, 0..) |node_value, index| {
        const node_object = node_value.object;
        const from_json = node_object.get("from").?.array;
        var from = try allocator.alloc(i64, from_json.items.len);
        for (from_json.items, 0..) |from_value, from_index| {
            from[from_index] = from_value.integer;
        }

        execution_nodes[index] = .{
            .index = @intCast(node_object.get("index").?.integer),
            .path = try allocator.dupe(u8, node_object.get("path").?.string),
            .kind = try allocator.dupe(u8, node_object.get("kind").?.string),
            .from = from,
        };
    }

    return .{
        .allocator = allocator,
        .format_version = root.get("format_version").?.integer,
        .model_name = try allocator.dupe(u8, root.get("model_name").?.string),
        .class_count = metadata.get("class_count").?.integer,
        .tensors = tensors,
        .execution_nodes = execution_nodes,
    };
}

pub fn loadSummary(allocator: std.mem.Allocator, graph_path: []const u8) !Summary {
    var model_graph = try load(allocator, graph_path);
    defer model_graph.deinit();

    return .{
        .format_version = model_graph.format_version,
        .model_name = try allocator.dupe(u8, model_graph.model_name),
        .tensor_count = model_graph.tensors.len,
        .execution_nodes = model_graph.execution_nodes.len,
        .class_count = model_graph.class_count,
    };
}
