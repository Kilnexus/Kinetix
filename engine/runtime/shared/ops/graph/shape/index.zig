const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub fn reshape(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
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

pub fn shapeOp(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const values = try allocator.alloc(i64, input.shape.len);
    defer allocator.free(values);
    for (input.shape, values) |dim, *slot| slot.* = @intCast(dim);
    return try Tensor.fromI64(allocator, &.{input.shape.len}, values);
}

pub fn flatten(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const axis = try normalizeFlattenAxis(common.attributeInt(node, "axis") orelse 1, input.shape.len);
    const out_shape = [_]usize{
        common.elementCountFromShape(input.shape[0..axis]),
        common.elementCountFromShape(input.shape[axis..]),
    };
    return try common.cloneWithShape(allocator, input, &out_shape);
}

fn normalizeFlattenAxis(axis: i64, rank: usize) !usize {
    const signed_rank: i64 = @intCast(rank);
    const normalized = if (axis < 0) axis + signed_rank else axis;
    if (normalized < 0 or normalized > signed_rank) return error.InvalidOperatorAttribute;
    return @intCast(normalized);
}

pub fn unsqueeze(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const axes = if (inputs.len == 2)
        try common.indicesToOwnedI64(allocator, inputs[1].*)
    else
        try common.attributeIntsOwned(allocator, node, "axes");
    defer allocator.free(axes);
    if (axes.len == 0) return error.MissingOperatorAttribute;

    const out_rank = input.shape.len + axes.len;
    const normalized = try common.normalizeAxesOwned(allocator, axes, out_rank);
    defer allocator.free(normalized);
    std.mem.sort(usize, normalized, {}, comptime std.sort.asc(usize));
    try common.ensureUniqueAxes(normalized);

    const out_shape = try allocator.alloc(usize, out_rank);
    defer allocator.free(out_shape);
    var in_index: usize = 0;
    for (out_shape, 0..) |*dim, out_index| {
        if (common.containsAxis(normalized, out_index)) {
            dim.* = 1;
        } else {
            dim.* = input.shape[in_index];
            in_index += 1;
        }
    }
    return try common.cloneWithShape(allocator, input, out_shape);
}

pub fn squeeze(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const raw_axes = if (inputs.len == 2)
        try common.indicesToOwnedI64(allocator, inputs[1].*)
    else
        try common.attributeIntsOwned(allocator, node, "axes");
    defer allocator.free(raw_axes);

    const axes = try common.normalizeAxesOwned(allocator, raw_axes, input.shape.len);
    defer allocator.free(axes);
    std.mem.sort(usize, axes, {}, comptime std.sort.asc(usize));
    try common.ensureUniqueAxes(axes);

    var out_shape = std.ArrayListUnmanaged(usize).empty;
    defer out_shape.deinit(allocator);
    for (input.shape, 0..) |dim, index| {
        const selected = if (axes.len == 0) dim == 1 else common.containsAxis(axes, index);
        if (selected) {
            if (dim != 1) return error.ShapeMismatch;
            continue;
        }
        try out_shape.append(allocator, dim);
    }
    return try common.cloneWithShape(allocator, input, out_shape.items);
}

pub fn concat(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len == 0) return error.InvalidOperatorArity;
    const first = inputs[0].*;
    if (first.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse return error.MissingOperatorAttribute, first.shape.len);
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

pub fn transpose(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
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

fn concatValues(comptime T: type, allocator: std.mem.Allocator, inputs: []const *const Tensor, out_shape: []const usize, axis: usize) !Tensor {
    const out = try allocator.alloc(T, common.elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const inner = common.elementCountFromShape(out_shape[axis + 1 ..]);
    const outer = common.elementCountFromShape(out_shape[0..axis]);
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
    const raw = try common.attributeIntsOwned(allocator, node, "perm");
    defer allocator.free(raw);
    if (raw.len == 0) {
        const out = try allocator.alloc(usize, rank);
        errdefer allocator.free(out);
        for (out, 0..) |*slot, index| slot.* = rank - 1 - index;
        return out;
    }
    if (raw.len != rank) return error.InvalidOperatorAttribute;
    const out = try common.normalizeAxesOwned(allocator, raw, rank);
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
    const out = try allocator.alloc(T, common.elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const input_strides = try common.stridesOwned(allocator, input_shape);
    defer allocator.free(input_strides);
    const out_strides = try common.stridesOwned(allocator, out_shape);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, out_shape.len);
    defer allocator.free(out_coords);
    const input_coords = try allocator.alloc(usize, input_shape.len);
    defer allocator.free(input_coords);

    for (out, 0..) |*slot, linear| {
        common.linearToCoords(linear, out_shape, out_strides, out_coords);
        for (perm, 0..) |input_axis, out_axis| input_coords[input_axis] = out_coords[out_axis];
        slot.* = values[common.coordsToLinear(input_coords, input_strides)];
    }
    return out;
}
