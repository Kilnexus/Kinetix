const std = @import("std");
const OpType = @import("../ops/op.zig").OpType;

pub const Node = struct {
    id: usize,
    name: []u8,
    op_type: OpType,
    inputs: []usize,
    output_shape: []usize,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node),

    pub fn init(allocator: std.mem.Allocator) Graph {
        return .{
            .allocator = allocator,
            .nodes = .{},
        };
    }

    pub fn deinit(self: *Graph) void {
        for (self.nodes.items) |node| {
            self.allocator.free(node.name);
            self.allocator.free(node.inputs);
            self.allocator.free(node.output_shape);
        }
        self.nodes.deinit(self.allocator);
    }

    pub fn addNode(
        self: *Graph,
        name: []const u8,
        op_type: OpType,
        inputs: []const usize,
        output_shape: []const usize,
    ) !usize {
        const id = self.nodes.items.len;
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const owned_inputs = try self.allocator.dupe(usize, inputs);
        errdefer self.allocator.free(owned_inputs);
        const owned_shape = try self.allocator.dupe(usize, output_shape);
        errdefer self.allocator.free(owned_shape);

        try self.nodes.append(self.allocator, .{
            .id = id,
            .name = owned_name,
            .op_type = op_type,
            .inputs = owned_inputs,
            .output_shape = owned_shape,
        });
        return id;
    }

    pub fn topoOrder(self: *const Graph, allocator: std.mem.Allocator) ![]usize {
        const node_count = self.nodes.items.len;
        var indegree = try allocator.alloc(usize, node_count);
        defer allocator.free(indegree);
        @memset(indegree, 0);

        for (self.nodes.items) |node| {
            indegree[node.id] = node.inputs.len;
        }

        var queue: std.ArrayList(usize) = .{};
        defer queue.deinit(allocator);
        for (0..node_count) |i| {
            if (indegree[i] == 0) {
                try queue.append(allocator, i);
            }
        }

        var order: std.ArrayList(usize) = .{};
        errdefer order.deinit(allocator);

        var read_idx: usize = 0;
        while (read_idx < queue.items.len) : (read_idx += 1) {
            const current = queue.items[read_idx];
            try order.append(allocator, current);

            for (self.nodes.items) |node| {
                if (contains(node.inputs, current)) {
                    indegree[node.id] -= 1;
                    if (indegree[node.id] == 0) {
                        try queue.append(allocator, node.id);
                    }
                }
            }
        }

        if (order.items.len != node_count) {
            return error.CycleDetected;
        }

        return order.toOwnedSlice(allocator);
    }

    fn contains(values: []const usize, needle: usize) bool {
        for (values) |v| {
            if (v == needle) return true;
        }
        return false;
    }
};

test "graph topo order linear dag" {
    var graph = Graph.init(std.testing.allocator);
    defer graph.deinit();

    const n0 = try graph.addNode("input", .input, &[_]usize{}, &[_]usize{ 1, 3, 32, 32 });
    const n1 = try graph.addNode("conv", .conv2d, &[_]usize{n0}, &[_]usize{ 1, 16, 32, 32 });
    _ = try graph.addNode("relu", .relu, &[_]usize{n1}, &[_]usize{ 1, 16, 32, 32 });

    const order = try graph.topoOrder(std.testing.allocator);
    defer std.testing.allocator.free(order);

    try std.testing.expectEqual(@as(usize, 3), order.len);
    try std.testing.expectEqual(@as(usize, 0), order[0]);
    try std.testing.expectEqual(@as(usize, 1), order[1]);
    try std.testing.expectEqual(@as(usize, 2), order[2]);
}
