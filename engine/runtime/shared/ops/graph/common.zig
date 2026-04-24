const std = @import("std");
const shared_graph = @import("shared_graph");
const onnx_metadata = shared_graph.onnx.metadata;
const tensor_mod = shared_graph.runtime.tensor;

pub const Tensor = tensor_mod.Tensor;

pub fn attributeInt(node: onnx_metadata.NodeInfo, name: []const u8) ?i64 {
    for (node.attributes) |attribute| {
        if (std.mem.eql(u8, attribute.name, name) and attribute.int_count != 0) return attribute.int_value;
    }
    return null;
}

pub fn attributeFloat(node: onnx_metadata.NodeInfo, name: []const u8) ?f32 {
    for (node.attributes) |attribute| {
        if (std.mem.eql(u8, attribute.name, name) and attribute.float_count != 0) return attribute.float_value;
    }
    return null;
}

pub fn attributeIntsOwned(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, name: []const u8) ![]i64 {
    for (node.attributes) |attribute| {
        if (!std.mem.eql(u8, attribute.name, name)) continue;
        if (attribute.int_values.len == 0) return try allocator.alloc(i64, 0);
        return try allocator.dupe(i64, attribute.int_values);
    }
    return try allocator.alloc(i64, 0);
}

pub fn indicesToOwnedI64(allocator: std.mem.Allocator, tensor: Tensor) ![]i64 {
    return switch (tensor.buffer) {
        .i64 => |values| try allocator.dupe(i64, values),
        .i32 => |values| blk: {
            const out = try allocator.alloc(i64, values.len);
            errdefer allocator.free(out);
            for (values, out) |value, *slot| slot.* = value;
            break :blk out;
        },
        else => error.UnsupportedTensorDType,
    };
}

pub fn normalizeAxis(axis: i64, rank: usize) !usize {
    if (rank == 0) return error.UnsupportedTensorRank;
    const signed_rank: i64 = @intCast(rank);
    const normalized = if (axis < 0) axis + signed_rank else axis;
    if (normalized < 0 or normalized >= signed_rank) return error.InvalidOperatorAttribute;
    return @intCast(normalized);
}

pub fn normalizeAxesOwned(allocator: std.mem.Allocator, axes: []const i64, rank: usize) ![]usize {
    const out = try allocator.alloc(usize, axes.len);
    errdefer allocator.free(out);
    for (axes, out) |axis, *slot| slot.* = try normalizeAxis(axis, rank);
    return out;
}

pub fn ensureUniqueAxes(axes: []const usize) !void {
    if (axes.len < 2) return;
    for (axes[1..], 1..) |axis, index| {
        if (axis == axes[index - 1]) return error.InvalidOperatorAttribute;
    }
}

pub fn containsAxis(axes: []const usize, needle: usize) bool {
    for (axes) |axis| if (axis == needle) return true;
    return false;
}

pub fn cloneWithShape(allocator: std.mem.Allocator, input: Tensor, shape_value: []const usize) !Tensor {
    if (elementCountFromShape(shape_value) != input.elementCount()) return error.ShapeMismatch;
    return switch (input.buffer) {
        .f32 => |values| try Tensor.fromF32(allocator, shape_value, values),
        .i32 => |values| try Tensor.fromI32(allocator, shape_value, values),
        .i64 => |values| try Tensor.fromI64(allocator, shape_value, values),
    };
}

pub fn elementCountFromShape(shape_value: []const usize) usize {
    var total: usize = 1;
    for (shape_value) |dim| total *= dim;
    return total;
}

pub fn stridesOwned(allocator: std.mem.Allocator, shape_value: []const usize) ![]usize {
    const strides = try allocator.alloc(usize, shape_value.len);
    errdefer allocator.free(strides);
    var stride: usize = 1;
    var index = shape_value.len;
    while (index > 0) {
        index -= 1;
        strides[index] = stride;
        stride *= shape_value[index];
    }
    return strides;
}

pub fn linearToCoords(linear: usize, shape_value: []const usize, strides: []const usize, coords: []usize) void {
    if (shape_value.len == 0) return;
    var remaining = linear;
    for (shape_value, strides, coords) |dim, stride, *coord| {
        if (dim == 0) {
            coord.* = 0;
        } else {
            coord.* = remaining / stride;
            remaining %= stride;
        }
    }
}

pub fn coordsToLinear(coords: []const usize, strides: []const usize) usize {
    var out: usize = 0;
    for (coords, strides) |coord, stride| out += coord * stride;
    return out;
}

pub fn dimsToShape(allocator: std.mem.Allocator, dims: []const onnx_metadata.Dimension) ![]usize {
    const out = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(out);
    for (dims, out) |dim, *slot| {
        slot.* = switch (dim) {
            .value => |value| if (value < 0) return error.DynamicTensorShape else @intCast(value),
            else => return error.DynamicTensorShape,
        };
    }
    return out;
}

pub fn defaultAxesI64(allocator: std.mem.Allocator, len: usize) ![]i64 {
    const out = try allocator.alloc(i64, len);
    errdefer allocator.free(out);
    for (out, 0..) |*slot, index| slot.* = @intCast(index);
    return out;
}

pub fn onesI64(allocator: std.mem.Allocator, len: usize) ![]i64 {
    const out = try allocator.alloc(i64, len);
    errdefer allocator.free(out);
    @memset(out, 1);
    return out;
}
