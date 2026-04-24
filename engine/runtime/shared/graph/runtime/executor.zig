const std = @import("std");
const onnx_metadata = @import("../onnx/metadata.zig");
const tensor_mod = @import("tensor.zig");
const graph_ops = @import("shared_ops").graph;

pub const Tensor = tensor_mod.Tensor;

pub const NamedTensor = struct {
    name: []const u8,
    tensor: Tensor,
};

pub const ExecutionResult = struct {
    allocator: std.mem.Allocator,
    outputs: []NamedTensor,

    pub fn deinit(self: *ExecutionResult) void {
        for (self.outputs) |*item| {
            self.allocator.free(item.name);
            item.tensor.deinit();
        }
        self.allocator.free(self.outputs);
        self.* = undefined;
    }
};

const ValueTable = struct {
    allocator: std.mem.Allocator,
    map: std.StringArrayHashMapUnmanaged(Tensor) = .empty,

    fn deinit(self: *ValueTable) void {
        for (self.map.keys()) |key| self.allocator.free(key);
        for (self.map.values()) |*value| value.deinit();
        self.map.deinit(self.allocator);
    }

    fn putOwned(self: *ValueTable, name: []const u8, tensor: Tensor) !void {
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        const result = try self.map.getOrPut(self.allocator, key);
        if (result.found_existing) {
            self.allocator.free(key);
            result.value_ptr.deinit();
        }
        result.key_ptr.* = key;
        result.value_ptr.* = tensor;
    }

    fn get(self: *const ValueTable, name: []const u8) ?*const Tensor {
        return self.map.getPtr(name);
    }
};

pub fn execute(
    allocator: std.mem.Allocator,
    graph: onnx_metadata.GraphMetadata,
    inputs: []const NamedTensor,
) !ExecutionResult {
    var table = ValueTable{ .allocator = allocator };
    defer table.deinit();

    for (inputs) |input| {
        try table.putOwned(input.name, try input.tensor.clone(allocator));
    }

    for (graph.nodes) |node| {
        if (node.outputs.len == 0) continue;
        var input_ptr_stack: [8]*const Tensor = undefined;
        const input_ptrs = if (node.inputs.len <= input_ptr_stack.len)
            input_ptr_stack[0..node.inputs.len]
        else
            try allocator.alloc(*const Tensor, node.inputs.len);
        defer if (node.inputs.len > input_ptr_stack.len) allocator.free(input_ptrs);

        for (node.inputs, input_ptrs) |name, *slot| {
            slot.* = table.get(name) orelse return error.TensorNotFound;
        }

        var output = try graph_ops.execute(allocator, node, input_ptrs);
        errdefer output.deinit();
        try table.putOwned(node.outputs[0], output);
    }

    const outputs = try allocator.alloc(NamedTensor, graph.outputs.len);
    var filled: usize = 0;
    errdefer {
        for (outputs[0..filled]) |*item| {
            allocator.free(item.name);
            item.tensor.deinit();
        }
        allocator.free(outputs);
    }

    for (graph.outputs, outputs) |output_info, *slot| {
        const value = table.get(output_info.name) orelse return error.TensorNotFound;
        slot.* = .{
            .name = try allocator.dupe(u8, output_info.name),
            .tensor = try value.clone(allocator),
        };
        filled += 1;
    }

    return .{ .allocator = allocator, .outputs = outputs };
}

test "runtime executor runs simple graph" {
    var graph = try onnx_metadata.parseModel(std.testing.allocator, try simpleModelBytes(std.testing.allocator));
    defer graph.deinit();

    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2 }, &.{ 1, 2 });
    defer input.deinit();
    var weight = try Tensor.fromF32(std.testing.allocator, &.{ 2, 1 }, &.{ 10, 20 });
    defer weight.deinit();

    var result = try execute(std.testing.allocator, graph.graph, &.{
        .{ .name = "x", .tensor = input },
        .{ .name = "w", .tensor = weight },
    });
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);
    try std.testing.expectEqualSlices(f32, &.{50}, result.outputs[0].tensor.buffer.f32);
}

test "runtime executor runs constant reshape graph" {
    var graph = try onnx_metadata.parseModel(std.testing.allocator, try reshapeModelBytes(std.testing.allocator));
    defer graph.deinit();

    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();

    var result = try execute(std.testing.allocator, graph.graph, &.{
        .{ .name = "x", .tensor = input },
    });
    defer result.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, result.outputs[0].tensor.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, result.outputs[0].tensor.buffer.f32);
}

fn simpleModelBytes(allocator: std.mem.Allocator) ![]u8 {
    var graph = std.ArrayList(u8).init(allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "simple");
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(allocator, "x", 1, &.{ 1, 2 }));
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(allocator, "w", 1, &.{ 2, 1 }));
    try appendOwnedMessageField(&graph, 12, try valueInfoMessage(allocator, "y", 1, &.{ 1, 1 }));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "matmul", "MatMul", &.{ "x", "w" }, &.{"y"}));

    var model = std.ArrayList(u8).init(allocator);
    errdefer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    return try model.toOwnedSlice();
}

fn reshapeModelBytes(allocator: std.mem.Allocator) ![]u8 {
    var graph = std.ArrayList(u8).init(allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "reshape");
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(allocator, "x", 1, &.{ 2, 2 }));
    try appendOwnedMessageField(&graph, 12, try valueInfoMessage(allocator, "y", 1, &.{ 1, 4 }));
    try appendOwnedMessageField(&graph, 1, try constantNodeMessage(allocator, "shape_const", "shape", &.{ 1, 4 }));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "reshape", "Reshape", &.{ "x", "shape" }, &.{"y"}));

    var model = std.ArrayList(u8).init(allocator);
    errdefer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    return try model.toOwnedSlice();
}

fn nodeMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    op_type: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (inputs) |input| try appendStringField(&out, 1, input);
    for (outputs) |output| try appendStringField(&out, 2, output);
    try appendStringField(&out, 3, name);
    try appendStringField(&out, 4, op_type);
    return try out.toOwnedSlice();
}

fn constantNodeMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    output: []const u8,
    values: []const i64,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 2, output);
    try appendStringField(&out, 3, name);
    try appendStringField(&out, 4, "Constant");
    try appendOwnedMessageField(&out, 5, try constantValueAttribute(allocator, values));
    return try out.toOwnedSlice();
}

fn constantValueAttribute(allocator: std.mem.Allocator, values: []const i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, "value");
    try appendOwnedMessageField(&out, 5, try int64TensorMessage(allocator, values));
    try appendVarintField(&out, 20, 4);
    return try out.toOwnedSlice();
}

fn int64TensorMessage(allocator: std.mem.Allocator, values: []const i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendVarintField(&out, 1, @intCast(values.len));
    try appendVarintField(&out, 2, 7);
    for (values) |value| try appendVarintField(&out, 7, @intCast(value));
    return try out.toOwnedSlice();
}

fn valueInfoMessage(allocator: std.mem.Allocator, name: []const u8, elem_type: u32, dims: []const i64) ![]u8 {
    var tensor_type = std.ArrayList(u8).init(allocator);
    defer tensor_type.deinit();
    try appendVarintField(&tensor_type, 1, elem_type);
    try appendOwnedMessageField(&tensor_type, 2, try shapeMessage(allocator, dims));

    var type_proto = std.ArrayList(u8).init(allocator);
    defer type_proto.deinit();
    try appendMessageField(&type_proto, 1, tensor_type.items);

    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, name);
    try appendMessageField(&out, 2, type_proto.items);
    return try out.toOwnedSlice();
}

fn shapeMessage(allocator: std.mem.Allocator, dims: []const i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (dims) |dim| {
        var dim_message = std.ArrayList(u8).init(allocator);
        defer dim_message.deinit();
        try appendVarintField(&dim_message, 1, @intCast(dim));
        try appendMessageField(&out, 1, dim_message.items);
    }
    return try out.toOwnedSlice();
}

fn appendOwnedMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []u8) !void {
    defer bytes.allocator.free(payload);
    try appendMessageField(bytes, field_number, payload);
}

fn appendMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, payload.len);
    try bytes.appendSlice(payload);
}

fn appendStringField(bytes: *std.ArrayList(u8), field_number: u64, value: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, value.len);
    try bytes.appendSlice(value);
}

fn appendVarintField(bytes: *std.ArrayList(u8), field_number: u64, value: u64) !void {
    try writeKey(bytes, field_number, 0);
    try writeVarint(bytes, value);
}

fn writeKey(bytes: *std.ArrayList(u8), field_number: u64, wire_type: u3) !void {
    try writeVarint(bytes, (field_number << 3) | wire_type);
}

fn writeVarint(bytes: *std.ArrayList(u8), raw: u64) !void {
    var value = raw;
    while (value >= 0x80) {
        try bytes.append(@intCast((value & 0x7f) | 0x80));
        value >>= 7;
    }
    try bytes.append(@intCast(value));
}
