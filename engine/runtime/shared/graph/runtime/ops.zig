const std = @import("std");
const onnx_metadata = @import("../onnx/metadata.zig");
const tensor_mod = @import("tensor.zig");

pub const Tensor = tensor_mod.Tensor;

pub fn isSupported(op_type: []const u8) bool {
    return std.mem.eql(u8, op_type, "Constant") or
        std.mem.eql(u8, op_type, "Identity") or
        std.mem.eql(u8, op_type, "Add") or
        std.mem.eql(u8, op_type, "Mul") or
        std.mem.eql(u8, op_type, "Relu") or
        std.mem.eql(u8, op_type, "Reshape") or
        std.mem.eql(u8, op_type, "MatMul") or
        std.mem.eql(u8, op_type, "Cast") or
        std.mem.eql(u8, op_type, "Shape") or
        std.mem.eql(u8, op_type, "Gather") or
        std.mem.eql(u8, op_type, "Unsqueeze") or
        std.mem.eql(u8, op_type, "Squeeze") or
        std.mem.eql(u8, op_type, "Concat") or
        std.mem.eql(u8, op_type, "Transpose");
}

pub fn execute(
    allocator: std.mem.Allocator,
    node: onnx_metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    if (std.mem.eql(u8, node.op_type, "Constant")) return try constant(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Identity")) return try identity(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Add")) return try elementwise(allocator, inputs, .add);
    if (std.mem.eql(u8, node.op_type, "Mul")) return try elementwise(allocator, inputs, .mul);
    if (std.mem.eql(u8, node.op_type, "Relu")) return try relu(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Reshape")) return try reshape(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "MatMul")) return try matmul(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Cast")) return try cast(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Shape")) return try shapeOp(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Gather")) return try gather(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Unsqueeze")) return try unsqueeze(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Squeeze")) return try squeeze(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Concat")) return try concat(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Transpose")) return try transpose(allocator, node, inputs);
    return error.UnsupportedOnnxOperator;
}

fn constant(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 0) return error.InvalidOperatorArity;
    for (node.attributes) |attribute| {
        if (!std.mem.eql(u8, attribute.name, "value")) continue;
        const literal = attribute.tensor orelse return error.UnsupportedConstantAttribute;
        if (literal.elem_type.raw == 7) {
            const tensor_shape = try dimsToShape(allocator, literal.dims);
            defer allocator.free(tensor_shape);
            return try Tensor.fromI64(allocator, tensor_shape, literal.int64_values);
        }
        return error.UnsupportedConstantTensorType;
    }
    return error.MissingConstantValue;
}

fn identity(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    return try inputs[0].clone(allocator);
}

const ElementwiseMode = enum { add, mul };

fn elementwise(allocator: std.mem.Allocator, inputs: []const *const Tensor, mode: ElementwiseMode) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const lhs = inputs[0].*;
    const rhs = inputs[1].*;
    if (!lhs.sameShape(rhs)) return error.ShapeMismatch;
    if (lhs.buffer != .f32 or rhs.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, lhs.buffer.f32.len);
    errdefer allocator.free(out);
    for (lhs.buffer.f32, rhs.buffer.f32, out) |a, b, *slot| {
        slot.* = switch (mode) {
            .add => a + b,
            .mul => a * b,
        };
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, lhs.shape),
        .buffer = .{ .f32 = out },
    };
}

fn relu(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| slot.* = @max(value, 0);
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

fn reshape(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const shape_tensor = inputs[1].*;
    if (shape_tensor.buffer != .i64) return error.UnsupportedTensorDType;
    const new_shape = try allocator.alloc(usize, shape_tensor.buffer.i64.len);
    defer allocator.free(new_shape);
    var inferred_index: ?usize = null;
    var known_product: usize = 1;
    for (shape_tensor.buffer.i64, new_shape, 0..) |dim, *slot, index| {
        if (dim == -1) {
            if (inferred_index != null) return error.InvalidTensorShape;
            inferred_index = index;
            slot.* = 1;
            continue;
        }
        if (dim < 0) return error.InvalidTensorShape;
        slot.* = @intCast(dim);
        known_product = try std.math.mul(usize, known_product, slot.*);
    }
    const element_count = input.elementCount();
    if (inferred_index) |index| {
        if (known_product == 0 or element_count % known_product != 0) return error.ShapeMismatch;
        new_shape[index] = element_count / known_product;
    } else if (known_product != element_count) {
        return error.ShapeMismatch;
    }
    return switch (input.buffer) {
        .f32 => |values| try Tensor.fromF32(allocator, new_shape, values),
        .i32 => |values| try Tensor.fromI32(allocator, new_shape, values),
        .i64 => |values| try Tensor.fromI64(allocator, new_shape, values),
    };
}

fn matmul(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const lhs = inputs[0].*;
    const rhs = inputs[1].*;
    if (lhs.buffer != .f32 or rhs.buffer != .f32) return error.UnsupportedTensorDType;
    if (lhs.shape.len != 2 or rhs.shape.len != 2) return error.UnsupportedTensorRank;
    const m = lhs.shape[0];
    const k = lhs.shape[1];
    if (rhs.shape[0] != k) return error.ShapeMismatch;
    const n = rhs.shape[1];
    const out = try allocator.alloc(f32, m * n);
    errdefer allocator.free(out);
    @memset(out, 0);
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |inner| {
                sum += lhs.buffer.f32[row * k + inner] * rhs.buffer.f32[inner * n + col];
            }
            out[row * n + col] = sum;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ m, n }),
        .buffer = .{ .f32 = out },
    };
}

fn cast(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const to = attributeInt(node, "to") orelse return error.MissingOperatorAttribute;
    const input = inputs[0].*;
    return switch (to) {
        1 => switch (input.buffer) {
            .f32 => |values| try Tensor.fromF32(allocator, input.shape, values),
            .i32 => |values| blk: {
                const out = try allocator.alloc(f32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @floatFromInt(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .f32 = out } };
            },
            .i64 => |values| blk: {
                const out = try allocator.alloc(f32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @floatFromInt(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .f32 = out } };
            },
        },
        6 => switch (input.buffer) {
            .i32 => |values| try Tensor.fromI32(allocator, input.shape, values),
            .i64 => |values| blk: {
                const out = try allocator.alloc(i32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intCast(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i32 = out } };
            },
            .f32 => |values| blk: {
                const out = try allocator.alloc(i32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intFromFloat(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i32 = out } };
            },
        },
        7 => switch (input.buffer) {
            .i64 => |values| try Tensor.fromI64(allocator, input.shape, values),
            .i32 => |values| blk: {
                const out = try allocator.alloc(i64, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = value;
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i64 = out } };
            },
            .f32 => |values| blk: {
                const out = try allocator.alloc(i64, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intFromFloat(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i64 = out } };
            },
        },
        else => error.UnsupportedTensorDType,
    };
}

fn shapeOp(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const values = try allocator.alloc(i64, input.shape.len);
    defer allocator.free(values);
    for (input.shape, values) |dim, *slot| slot.* = @intCast(dim);
    return try Tensor.fromI64(allocator, &.{input.shape.len}, values);
}

fn gather(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const data = inputs[0].*;
    const indices = inputs[1].*;
    if (data.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try normalizeAxis(attributeInt(node, "axis") orelse 0, data.shape.len);
    const index_values = try indicesToOwnedI64(allocator, indices);
    defer allocator.free(index_values);

    const out_rank = data.shape.len + indices.shape.len - 1;
    const out_shape = try allocator.alloc(usize, out_rank);
    defer allocator.free(out_shape);
    @memcpy(out_shape[0..axis], data.shape[0..axis]);
    @memcpy(out_shape[axis .. axis + indices.shape.len], indices.shape);
    @memcpy(out_shape[axis + indices.shape.len ..], data.shape[axis + 1 ..]);

    return switch (data.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .f32 = try gatherValues(f32, allocator, values, data.shape, indices.shape, index_values, axis, out_shape) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i32 = try gatherValues(i32, allocator, values, data.shape, indices.shape, index_values, axis, out_shape) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i64 = try gatherValues(i64, allocator, values, data.shape, indices.shape, index_values, axis, out_shape) } },
    };
}

fn unsqueeze(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const axes = if (inputs.len == 2)
        try indicesToOwnedI64(allocator, inputs[1].*)
    else
        try attributeIntsOwned(allocator, node, "axes");
    defer allocator.free(axes);
    if (axes.len == 0) return error.MissingOperatorAttribute;

    const out_rank = input.shape.len + axes.len;
    const normalized = try normalizeAxesOwned(allocator, axes, out_rank);
    defer allocator.free(normalized);
    std.mem.sort(usize, normalized, {}, comptime std.sort.asc(usize));
    try ensureUniqueAxes(normalized);

    const out_shape = try allocator.alloc(usize, out_rank);
    defer allocator.free(out_shape);
    var in_index: usize = 0;
    for (out_shape, 0..) |*dim, out_index| {
        if (containsAxis(normalized, out_index)) {
            dim.* = 1;
        } else {
            dim.* = input.shape[in_index];
            in_index += 1;
        }
    }
    return try cloneWithShape(allocator, input, out_shape);
}

fn squeeze(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const raw_axes = if (inputs.len == 2)
        try indicesToOwnedI64(allocator, inputs[1].*)
    else
        try attributeIntsOwned(allocator, node, "axes");
    defer allocator.free(raw_axes);

    const axes = try normalizeAxesOwned(allocator, raw_axes, input.shape.len);
    defer allocator.free(axes);
    std.mem.sort(usize, axes, {}, comptime std.sort.asc(usize));
    try ensureUniqueAxes(axes);

    var out_shape = std.ArrayListUnmanaged(usize).empty;
    defer out_shape.deinit(allocator);
    for (input.shape, 0..) |dim, index| {
        const selected = if (axes.len == 0) dim == 1 else containsAxis(axes, index);
        if (selected) {
            if (dim != 1) return error.ShapeMismatch;
            continue;
        }
        try out_shape.append(allocator, dim);
    }
    return try cloneWithShape(allocator, input, out_shape.items);
}

fn concat(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len == 0) return error.InvalidOperatorArity;
    const first = inputs[0].*;
    if (first.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try normalizeAxis(attributeInt(node, "axis") orelse return error.MissingOperatorAttribute, first.shape.len);
    const dtype = first.dtype();

    const out_shape = try allocator.dupe(usize, first.shape);
    defer allocator.free(out_shape);
    out_shape[axis] = 0;
    for (inputs) |input_ref| {
        const input = input_ref.*;
        if (input.dtype() != dtype or input.shape.len != first.shape.len) return error.ShapeMismatch;
        for (input.shape, 0..) |dim, index| {
            if (index == axis) continue;
            if (dim != first.shape[index]) return error.ShapeMismatch;
        }
        out_shape[axis] += input.shape[axis];
    }

    return switch (dtype) {
        .f32 => try concatValues(f32, allocator, inputs, out_shape, axis),
        .i32 => try concatValues(i32, allocator, inputs, out_shape, axis),
        .i64 => try concatValues(i64, allocator, inputs, out_shape, axis),
    };
}

fn transpose(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const perm = try permutationOwned(allocator, node, input.shape.len);
    defer allocator.free(perm);
    const out_shape = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(out_shape);
    for (perm, out_shape) |axis, *dim| dim.* = input.shape[axis];

    return switch (input.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .f32 = try transposeValues(f32, allocator, values, input.shape, out_shape, perm) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i32 = try transposeValues(i32, allocator, values, input.shape, out_shape, perm) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i64 = try transposeValues(i64, allocator, values, input.shape, out_shape, perm) } },
    };
}

fn attributeInt(node: onnx_metadata.NodeInfo, name: []const u8) ?i64 {
    for (node.attributes) |attribute| {
        if (std.mem.eql(u8, attribute.name, name) and attribute.int_count != 0) return attribute.int_value;
    }
    return null;
}

fn attributeIntsOwned(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, name: []const u8) ![]i64 {
    for (node.attributes) |attribute| {
        if (!std.mem.eql(u8, attribute.name, name)) continue;
        if (attribute.int_values.len == 0) return try allocator.alloc(i64, 0);
        return try allocator.dupe(i64, attribute.int_values);
    }
    return try allocator.alloc(i64, 0);
}

fn indicesToOwnedI64(allocator: std.mem.Allocator, tensor: Tensor) ![]i64 {
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

fn normalizeAxis(axis: i64, rank: usize) !usize {
    if (rank == 0) return error.UnsupportedTensorRank;
    const signed_rank: i64 = @intCast(rank);
    const normalized = if (axis < 0) axis + signed_rank else axis;
    if (normalized < 0 or normalized >= signed_rank) return error.InvalidOperatorAttribute;
    return @intCast(normalized);
}

fn normalizeAxesOwned(allocator: std.mem.Allocator, axes: []const i64, rank: usize) ![]usize {
    const out = try allocator.alloc(usize, axes.len);
    errdefer allocator.free(out);
    for (axes, out) |axis, *slot| slot.* = try normalizeAxis(axis, rank);
    return out;
}

fn ensureUniqueAxes(axes: []const usize) !void {
    if (axes.len < 2) return;
    for (axes[1..], 1..) |axis, index| {
        if (axis == axes[index - 1]) return error.InvalidOperatorAttribute;
    }
}

fn containsAxis(axes: []const usize, needle: usize) bool {
    for (axes) |axis| if (axis == needle) return true;
    return false;
}

fn cloneWithShape(allocator: std.mem.Allocator, input: Tensor, shape_value: []const usize) !Tensor {
    if (elementCountFromShape(shape_value) != input.elementCount()) return error.ShapeMismatch;
    return switch (input.buffer) {
        .f32 => |values| try Tensor.fromF32(allocator, shape_value, values),
        .i32 => |values| try Tensor.fromI32(allocator, shape_value, values),
        .i64 => |values| try Tensor.fromI64(allocator, shape_value, values),
    };
}

fn elementCountFromShape(shape_value: []const usize) usize {
    var total: usize = 1;
    for (shape_value) |dim| total *= dim;
    return total;
}

fn stridesOwned(allocator: std.mem.Allocator, shape_value: []const usize) ![]usize {
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

fn linearToCoords(linear: usize, shape_value: []const usize, strides: []const usize, coords: []usize) void {
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

fn coordsToLinear(coords: []const usize, strides: []const usize) usize {
    var out: usize = 0;
    for (coords, strides) |coord, stride| out += coord * stride;
    return out;
}

fn gatherValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    data: []const T,
    data_shape: []const usize,
    indices_shape: []const usize,
    indices: []const i64,
    axis: usize,
    out_shape: []const usize,
) ![]T {
    const out = try allocator.alloc(T, elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const data_strides = try stridesOwned(allocator, data_shape);
    defer allocator.free(data_strides);
    const indices_strides = try stridesOwned(allocator, indices_shape);
    defer allocator.free(indices_strides);
    const out_strides = try stridesOwned(allocator, out_shape);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, out_shape.len);
    defer allocator.free(out_coords);
    const index_coords = try allocator.alloc(usize, indices_shape.len);
    defer allocator.free(index_coords);
    const data_coords = try allocator.alloc(usize, data_shape.len);
    defer allocator.free(data_coords);

    for (out, 0..) |*slot, linear| {
        linearToCoords(linear, out_shape, out_strides, out_coords);
        @memcpy(data_coords[0..axis], out_coords[0..axis]);
        @memcpy(index_coords, out_coords[axis .. axis + indices_shape.len]);
        @memcpy(data_coords[axis + 1 ..], out_coords[axis + indices_shape.len ..]);
        const raw_index = indices[coordsToLinear(index_coords, indices_strides)];
        const normalized = if (raw_index < 0) raw_index + @as(i64, @intCast(data_shape[axis])) else raw_index;
        if (normalized < 0 or normalized >= @as(i64, @intCast(data_shape[axis]))) return error.InvalidTensorIndex;
        data_coords[axis] = @intCast(normalized);
        slot.* = data[coordsToLinear(data_coords, data_strides)];
    }
    return out;
}

fn concatValues(comptime T: type, allocator: std.mem.Allocator, inputs: []const *const Tensor, out_shape: []const usize, axis: usize) !Tensor {
    const out = try allocator.alloc(T, elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const inner = elementCountFromShape(out_shape[axis + 1 ..]);
    const outer = elementCountFromShape(out_shape[0..axis]);
    var write_index: usize = 0;
    for (0..outer) |outer_index| {
        for (inputs) |input_ref| {
            const input = input_ref.*;
            const values = switch (input.buffer) {
                .f32 => |items| if (T == f32) items else return error.UnsupportedTensorDType,
                .i32 => |items| if (T == i32) items else return error.UnsupportedTensorDType,
                .i64 => |items| if (T == i64) items else return error.UnsupportedTensorDType,
            };
            const chunk_len = input.shape[axis] * inner;
            const read_start = outer_index * chunk_len;
            @memcpy(out[write_index .. write_index + chunk_len], values[read_start .. read_start + chunk_len]);
            write_index += chunk_len;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, out_shape),
        .buffer = switch (T) {
            f32 => .{ .f32 = out },
            i32 => .{ .i32 = out },
            i64 => .{ .i64 = out },
            else => unreachable,
        },
    };
}

fn permutationOwned(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, rank: usize) ![]usize {
    const raw = try attributeIntsOwned(allocator, node, "perm");
    defer allocator.free(raw);
    if (raw.len == 0) {
        const out = try allocator.alloc(usize, rank);
        errdefer allocator.free(out);
        for (out, 0..) |*slot, index| slot.* = rank - 1 - index;
        return out;
    }
    if (raw.len != rank) return error.InvalidOperatorAttribute;
    const out = try normalizeAxesOwned(allocator, raw, rank);
    errdefer allocator.free(out);
    const sorted = try allocator.dupe(usize, out);
    defer allocator.free(sorted);
    std.mem.sort(usize, sorted, {}, comptime std.sort.asc(usize));
    for (sorted, 0..) |axis, expected| {
        if (axis != expected) return error.InvalidOperatorAttribute;
    }
    return out;
}

fn transposeValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    input_shape: []const usize,
    out_shape: []const usize,
    perm: []const usize,
) ![]T {
    const out = try allocator.alloc(T, elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const input_strides = try stridesOwned(allocator, input_shape);
    defer allocator.free(input_strides);
    const out_strides = try stridesOwned(allocator, out_shape);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, out_shape.len);
    defer allocator.free(out_coords);
    const input_coords = try allocator.alloc(usize, input_shape.len);
    defer allocator.free(input_coords);

    for (out, 0..) |*slot, linear| {
        linearToCoords(linear, out_shape, out_strides, out_coords);
        for (perm, 0..) |input_axis, out_axis| input_coords[input_axis] = out_coords[out_axis];
        slot.* = values[coordsToLinear(input_coords, input_strides)];
    }
    return out;
}

fn dimsToShape(allocator: std.mem.Allocator, dims: []const onnx_metadata.Dimension) ![]usize {
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

test "runtime ops execute f32 matmul and relu" {
    var lhs = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    var rhs = try Tensor.fromF32(std.testing.allocator, &.{ 2, 1 }, &.{ 10, 20 });
    defer rhs.deinit();
    var out = try matmul(std.testing.allocator, &.{ &lhs, &rhs });
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 110 }, out.buffer.f32);

    var relu_out = try relu(std.testing.allocator, &.{&out});
    defer relu_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 110 }, relu_out.buffer.f32);
}

test "runtime ops execute shape cast and gather" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var shape_out = try shapeOp(std.testing.allocator, &.{&input});
    defer shape_out.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, shape_out.buffer.i64);

    var to_f32_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("to", 1)};
    const cast_node = testNode("Cast", to_f32_attr[0..]);
    var cast_out = try cast(std.testing.allocator, cast_node, &.{&shape_out});
    defer cast_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, cast_out.buffer.f32);

    var data = try Tensor.fromI64(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer data.deinit();
    var indices = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 2, 0 });
    defer indices.deinit();
    var axis_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("axis", 0)};
    const gather_node = testNode("Gather", axis_attr[0..]);
    var gather_out = try gather(std.testing.allocator, gather_node, &.{ &data, &indices });
    defer gather_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, gather_out.shape);
    try std.testing.expectEqualSlices(i64, &.{ 30, 10 }, gather_out.buffer.i64);
}

test "runtime ops execute unsqueeze squeeze concat and transpose" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var unsqueeze_axes_values = [_]i64{0};
    var unsqueeze_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("axes", unsqueeze_axes_values[0..])};
    const unsqueeze_node = testNode("Unsqueeze", unsqueeze_attrs[0..]);
    var unsqueezed = try unsqueeze(std.testing.allocator, unsqueeze_node, &.{&input});
    defer unsqueezed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, unsqueezed.shape);

    var squeeze_axes_values = [_]i64{0};
    var squeeze_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("axes", squeeze_axes_values[0..])};
    const squeeze_node = testNode("Squeeze", squeeze_attrs[0..]);
    var squeezed = try squeeze(std.testing.allocator, squeeze_node, &.{&unsqueezed});
    defer squeezed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, squeezed.shape);
    try std.testing.expectEqualSlices(f32, input.buffer.f32, squeezed.buffer.f32);

    var left = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer left.deinit();
    var right = try Tensor.fromI64(std.testing.allocator, &.{3}, &.{ 3, 4, 5 });
    defer right.deinit();
    var concat_axis_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("axis", 0)};
    const concat_node = testNode("Concat", concat_axis_attr[0..]);
    var concat_out = try concat(std.testing.allocator, concat_node, &.{ &left, &right });
    defer concat_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{5}, concat_out.shape);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, concat_out.buffer.i64);

    var perm_values = [_]i64{ 1, 0 };
    var transpose_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("perm", perm_values[0..])};
    const transpose_node = testNode("Transpose", transpose_attrs[0..]);
    var transposed = try transpose(std.testing.allocator, transpose_node, &.{&input});
    defer transposed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, transposed.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, transposed.buffer.f32);
}

fn testNode(op_type: []const u8, attributes: []onnx_metadata.AttributeInfo) onnx_metadata.NodeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(""),
        .op_type = @constCast(op_type),
        .domain = @constCast(""),
        .attributes = attributes,
    };
}

fn testIntAttribute(name: []const u8, value: i64) onnx_metadata.AttributeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(name),
        .int_value = value,
        .int_count = 1,
    };
}

fn testIntsAttribute(name: []const u8, values: []i64) onnx_metadata.AttributeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(name),
        .int_values = values,
        .int_count = values.len,
    };
}
