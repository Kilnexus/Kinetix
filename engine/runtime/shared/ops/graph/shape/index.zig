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

pub fn pad(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 2 or inputs.len > 4) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const pads = try common.indicesToOwnedI64(allocator, inputs[1].*);
    defer allocator.free(pads);
    if (pads.len != input.shape.len * 2) return error.InvalidOperatorAttribute;
    const mode = stringAttribute(node, "mode") orelse "constant";
    if (!std.mem.eql(u8, mode, "constant")) return error.UnsupportedOperatorAttribute;

    const constant_value = if (inputs.len >= 3) try scalarConstant(inputs[2].*) else 0;
    const out_shape = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(out_shape);
    const starts = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(starts);
    for (input.shape, out_shape, starts, 0..) |dim, *out_dim, *start, axis| {
        const before = pads[axis];
        const after = pads[axis + input.shape.len];
        if (before < 0 or after < 0) return error.UnsupportedOperatorAttribute;
        start.* = @intCast(before);
        out_dim.* = dim + @as(usize, @intCast(before)) + @as(usize, @intCast(after));
    }

    return switch (input.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .f32 = try padValues(f32, allocator, values, input.shape, out_shape, starts, @floatCast(constant_value)) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i32 = try padValues(i32, allocator, values, input.shape, out_shape, starts, @intFromFloat(constant_value)) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i64 = try padValues(i64, allocator, values, input.shape, out_shape, starts, @intFromFloat(constant_value)) } },
    };
}

pub fn expand(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const shape_tensor = inputs[1].*;
    const raw_shape = try common.indicesToOwnedI64(allocator, shape_tensor);
    defer allocator.free(raw_shape);
    const out_shape = try allocator.alloc(usize, raw_shape.len);
    defer allocator.free(out_shape);
    for (raw_shape, out_shape) |dim, *slot| {
        if (dim < 0) return error.InvalidTensorShape;
        slot.* = @intCast(dim);
    }
    if (out_shape.len < input.shape.len) return error.ShapeMismatch;

    return switch (input.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .f32 = try expandValues(f32, allocator, values, input.shape, out_shape) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i32 = try expandValues(i32, allocator, values, input.shape, out_shape) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i64 = try expandValues(i64, allocator, values, input.shape, out_shape) } },
    };
}

pub fn split(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    const index = common.attributeInt(node, "kinetix_output_index") orelse 0;
    return try splitOutput(allocator, node, inputs, @intCast(index));
}

pub fn splitOutput(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor, output_index: usize) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse 0, input.shape.len);
    const split_sizes = if (inputs.len == 2)
        try common.indicesToOwnedI64(allocator, inputs[1].*)
    else
        try common.attributeIntsOwned(allocator, node, "split");
    defer allocator.free(split_sizes);
    if (split_sizes.len == 0) return error.MissingOperatorAttribute;
    if (output_index >= split_sizes.len) return error.InvalidTensorIndex;
    var offset: usize = 0;
    for (split_sizes[0..output_index]) |size| {
        if (size < 0) return error.InvalidTensorShape;
        offset += @intCast(size);
    }
    const selected_size = split_sizes[output_index];
    if (selected_size < 0) return error.InvalidTensorShape;
    if (offset + @as(usize, @intCast(selected_size)) > input.shape[axis]) return error.ShapeMismatch;

    const out_shape = try allocator.dupe(usize, input.shape);
    defer allocator.free(out_shape);
    out_shape[axis] = @intCast(selected_size);
    return switch (input.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .f32 = try splitValues(f32, allocator, values, input.shape, axis, offset, out_shape) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i32 = try splitValues(i32, allocator, values, input.shape, axis, offset, out_shape) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, out_shape), .buffer = .{ .i64 = try splitValues(i64, allocator, values, input.shape, axis, offset, out_shape) } },
    };
}

pub fn resize(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    _ = node;
    if (inputs.len < 3 or inputs.len > 4) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len != 4) return error.UnsupportedTensorRank;
    const out_shape = if (inputs.len == 4)
        try resizeShapeFromSizes(allocator, inputs[3].*)
    else
        try resizeShapeFromScales(allocator, input.shape, inputs[2].*);
    defer allocator.free(out_shape);
    if (out_shape.len != 4) return error.UnsupportedTensorRank;
    if (out_shape[0] != input.shape[0] or out_shape[1] != input.shape[1]) return error.UnsupportedOperatorAttribute;

    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, out_shape),
        .buffer = .{ .f32 = try resizeNearestNchw(allocator, input.buffer.f32, input.shape, out_shape) },
    };
}

fn scalarConstant(tensor: Tensor) !f64 {
    if (tensor.elementCount() != 1) return error.ShapeMismatch;
    return switch (tensor.buffer) {
        .f32 => |values| values[0],
        .i32 => |values| @floatFromInt(values[0]),
        .i64 => |values| @floatFromInt(values[0]),
    };
}

fn resizeShapeFromSizes(allocator: std.mem.Allocator, tensor: Tensor) ![]usize {
    const values = try common.indicesToOwnedI64(allocator, tensor);
    defer allocator.free(values);
    const out = try allocator.alloc(usize, values.len);
    errdefer allocator.free(out);
    for (values, out) |value, *slot| {
        if (value <= 0) return error.InvalidTensorShape;
        slot.* = @intCast(value);
    }
    return out;
}

fn resizeShapeFromScales(allocator: std.mem.Allocator, input_shape: []const usize, tensor: Tensor) ![]usize {
    if (tensor.buffer != .f32) return error.UnsupportedTensorDType;
    if (tensor.buffer.f32.len != input_shape.len) return error.ShapeMismatch;
    const out = try allocator.alloc(usize, input_shape.len);
    errdefer allocator.free(out);
    for (input_shape, tensor.buffer.f32, out) |dim, scale, *slot| {
        if (scale <= 0) return error.InvalidTensorShape;
        slot.* = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(@as(f32, @floatFromInt(dim)) * scale))));
    }
    return out;
}

fn stringAttribute(node: onnx_metadata.NodeInfo, name: []const u8) ?[]const u8 {
    _ = node;
    _ = name;
    return null;
}

fn padValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    input_shape: []const usize,
    out_shape: []const usize,
    starts: []const usize,
    constant: T,
) ![]T {
    const out = try allocator.alloc(T, common.elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    @memset(out, constant);
    const input_strides = try common.stridesOwned(allocator, input_shape);
    defer allocator.free(input_strides);
    const out_strides = try common.stridesOwned(allocator, out_shape);
    defer allocator.free(out_strides);
    const input_coords = try allocator.alloc(usize, input_shape.len);
    defer allocator.free(input_coords);
    const out_coords = try allocator.alloc(usize, out_shape.len);
    defer allocator.free(out_coords);
    for (values, 0..) |value, linear| {
        common.linearToCoords(linear, input_shape, input_strides, input_coords);
        for (input_coords, out_coords, starts) |coord, *out_coord, start| out_coord.* = coord + start;
        out[common.coordsToLinear(out_coords, out_strides)] = value;
    }
    return out;
}

fn expandValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    input_shape: []const usize,
    out_shape: []const usize,
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
    const offset = out_shape.len - input_shape.len;
    for (out, 0..) |*slot, linear| {
        common.linearToCoords(linear, out_shape, out_strides, out_coords);
        for (input_shape, input_coords, 0..) |dim, *coord, axis| {
            const out_coord = out_coords[offset + axis];
            if (dim == 1) {
                coord.* = 0;
            } else if (dim == out_shape[offset + axis]) {
                coord.* = out_coord;
            } else {
                return error.ShapeMismatch;
            }
        }
        slot.* = values[common.coordsToLinear(input_coords, input_strides)];
    }
    return out;
}

fn splitValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    input_shape: []const usize,
    axis: usize,
    offset: usize,
    out_shape: []const usize,
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
        @memcpy(input_coords, out_coords);
        input_coords[axis] += offset;
        slot.* = values[common.coordsToLinear(input_coords, input_strides)];
    }
    return out;
}

fn resizeNearestNchw(
    allocator: std.mem.Allocator,
    values: []const f32,
    input_shape: []const usize,
    out_shape: []const usize,
) ![]f32 {
    const out = try allocator.alloc(f32, common.elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const in_n = input_shape[0];
    const in_c = input_shape[1];
    const in_h = input_shape[2];
    const in_w = input_shape[3];
    const out_h = out_shape[2];
    const out_w = out_shape[3];
    for (0..in_n) |n| {
        for (0..in_c) |c| {
            for (0..out_h) |oh| {
                const ih = @min(in_h - 1, (oh * in_h) / out_h);
                for (0..out_w) |ow| {
                    const iw = @min(in_w - 1, (ow * in_w) / out_w);
                    const out_index = ((n * in_c + c) * out_h + oh) * out_w + ow;
                    const in_index = ((n * in_c + c) * in_h + ih) * in_w + iw;
                    out[out_index] = values[in_index];
                }
            }
        }
    }
    return out;
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
