const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub fn gather(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const data = inputs[0].*;
    const indices = inputs[1].*;
    if (data.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse 0, data.shape.len);
    const index_values = try common.indicesToOwnedI64(allocator, indices);
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

pub fn slice(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    _ = node;
    if (inputs.len < 3 or inputs.len > 5) return error.InvalidOperatorArity;
    const data = inputs[0].*;
    const starts = try common.indicesToOwnedI64(allocator, inputs[1].*);
    defer allocator.free(starts);
    const ends = try common.indicesToOwnedI64(allocator, inputs[2].*);
    defer allocator.free(ends);
    if (starts.len != ends.len) return error.ShapeMismatch;

    const axes = if (inputs.len >= 4)
        try common.indicesToOwnedI64(allocator, inputs[3].*)
    else
        try common.defaultAxesI64(allocator, starts.len);
    defer allocator.free(axes);
    const steps = if (inputs.len >= 5)
        try common.indicesToOwnedI64(allocator, inputs[4].*)
    else
        try common.onesI64(allocator, starts.len);
    defer allocator.free(steps);
    if (axes.len != starts.len or steps.len != starts.len) return error.ShapeMismatch;

    const spec = try sliceSpecOwned(allocator, data.shape, starts, ends, axes, steps);
    defer {
        allocator.free(spec.starts);
        allocator.free(spec.steps);
        allocator.free(spec.shape);
    }

    return switch (data.buffer) {
        .f32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, spec.shape), .buffer = .{ .f32 = try sliceValues(f32, allocator, values, data.shape, spec.starts, spec.steps, spec.shape) } },
        .i32 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, spec.shape), .buffer = .{ .i32 = try sliceValues(i32, allocator, values, data.shape, spec.starts, spec.steps, spec.shape) } },
        .i64 => |values| Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, spec.shape), .buffer = .{ .i64 = try sliceValues(i64, allocator, values, data.shape, spec.starts, spec.steps, spec.shape) } },
    };
}

pub fn argMax(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len == 0) return error.UnsupportedTensorRank;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse 0, input.shape.len);
    const keepdims = (common.attributeInt(node, "keepdims") orelse 1) != 0;
    const select_last_index = (common.attributeInt(node, "select_last_index") orelse 0) != 0;

    var out_shape_list = std.ArrayListUnmanaged(usize).empty;
    defer out_shape_list.deinit(allocator);
    for (input.shape, 0..) |dim, index| {
        if (index == axis) {
            if (keepdims) try out_shape_list.append(allocator, 1);
        } else {
            try out_shape_list.append(allocator, dim);
        }
    }
    if (out_shape_list.items.len == 0) try out_shape_list.append(allocator, 1);

    const axis_dim = input.shape[axis];
    const inner = common.elementCountFromShape(input.shape[axis + 1 ..]);
    const outer = common.elementCountFromShape(input.shape[0..axis]);
    const out = try allocator.alloc(i64, common.elementCountFromShape(out_shape_list.items));
    errdefer allocator.free(out);
    var write_index: usize = 0;
    for (0..outer) |outer_index| {
        for (0..inner) |inner_index| {
            var best_index: usize = 0;
            var best_value = input.buffer.f32[outer_index * axis_dim * inner + inner_index];
            for (1..axis_dim) |axis_index| {
                const value = input.buffer.f32[outer_index * axis_dim * inner + axis_index * inner + inner_index];
                if (value > best_value or (select_last_index and value == best_value)) {
                    best_value = value;
                    best_index = axis_index;
                }
            }
            out[write_index] = @intCast(best_index);
            write_index += 1;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, out_shape_list.items),
        .buffer = .{ .i64 = out },
    };
}

pub fn nonZero(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const rank = input.shape.len;
    const count = countNonZero(input);
    const out = try allocator.alloc(i64, rank * count);
    errdefer allocator.free(out);
    const strides = try common.stridesOwned(allocator, input.shape);
    defer allocator.free(strides);
    const coords = try allocator.alloc(usize, rank);
    defer allocator.free(coords);
    var write_col: usize = 0;
    for (0..input.elementCount()) |linear| {
        if (!isNonZero(input, linear)) continue;
        common.linearToCoords(linear, input.shape, strides, coords);
        for (coords, 0..) |coord, axis| out[axis * count + write_col] = @intCast(coord);
        write_col += 1;
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ rank, count }),
        .buffer = .{ .i64 = out },
    };
}

fn countNonZero(input: Tensor) usize {
    var count: usize = 0;
    for (0..input.elementCount()) |index| {
        if (isNonZero(input, index)) count += 1;
    }
    return count;
}

fn isNonZero(input: Tensor, index: usize) bool {
    return switch (input.buffer) {
        .f32 => |values| values[index] != 0,
        .i32 => |values| values[index] != 0,
        .i64 => |values| values[index] != 0,
    };
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
    const out = try allocator.alloc(T, common.elementCountFromShape(out_shape));
    errdefer allocator.free(out);
    const data_strides = try common.stridesOwned(allocator, data_shape);
    defer allocator.free(data_strides);
    const indices_strides = try common.stridesOwned(allocator, indices_shape);
    defer allocator.free(indices_strides);
    const out_strides = try common.stridesOwned(allocator, out_shape);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, out_shape.len);
    defer allocator.free(out_coords);
    const index_coords = try allocator.alloc(usize, indices_shape.len);
    defer allocator.free(index_coords);
    const data_coords = try allocator.alloc(usize, data_shape.len);
    defer allocator.free(data_coords);

    for (out, 0..) |*slot, linear| {
        common.linearToCoords(linear, out_shape, out_strides, out_coords);
        @memcpy(data_coords[0..axis], out_coords[0..axis]);
        @memcpy(index_coords, out_coords[axis .. axis + indices_shape.len]);
        @memcpy(data_coords[axis + 1 ..], out_coords[axis + indices_shape.len ..]);
        const raw_index = indices[common.coordsToLinear(index_coords, indices_strides)];
        const normalized = if (raw_index < 0) raw_index + @as(i64, @intCast(data_shape[axis])) else raw_index;
        if (normalized < 0 or normalized >= @as(i64, @intCast(data_shape[axis]))) return error.InvalidTensorIndex;
        data_coords[axis] = @intCast(normalized);
        slot.* = data[common.coordsToLinear(data_coords, data_strides)];
    }
    return out;
}

const SliceSpec = struct {
    starts: []usize,
    steps: []usize,
    shape: []usize,
};

fn sliceSpecOwned(
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    starts: []const i64,
    ends: []const i64,
    axes: []const i64,
    steps: []const i64,
) !SliceSpec {
    const start_by_axis = try allocator.alloc(usize, input_shape.len);
    errdefer allocator.free(start_by_axis);
    const step_by_axis = try allocator.alloc(usize, input_shape.len);
    errdefer allocator.free(step_by_axis);
    const out_shape = try allocator.dupe(usize, input_shape);
    errdefer allocator.free(out_shape);
    @memset(start_by_axis, 0);
    @memset(step_by_axis, 1);

    for (starts, ends, axes, steps) |raw_start, raw_end, raw_axis, raw_step| {
        if (raw_step <= 0) return error.UnsupportedOperatorAttribute;
        const axis = try common.normalizeAxis(raw_axis, input_shape.len);
        const dim: i64 = @intCast(input_shape[axis]);
        const start = normalizeSliceBound(raw_start, dim);
        const end = normalizeSliceBound(raw_end, dim);
        const step: usize = @intCast(raw_step);
        start_by_axis[axis] = @intCast(start);
        step_by_axis[axis] = step;
        if (end <= start) {
            out_shape[axis] = 0;
        } else {
            out_shape[axis] = @intCast(@divTrunc(end - start + raw_step - 1, raw_step));
        }
    }
    return .{
        .starts = start_by_axis,
        .steps = step_by_axis,
        .shape = out_shape,
    };
}

fn normalizeSliceBound(raw: i64, dim: i64) i64 {
    var value = if (raw < 0) raw + dim else raw;
    if (value < 0) value = 0;
    if (value > dim) value = dim;
    return value;
}

fn sliceValues(
    comptime T: type,
    allocator: std.mem.Allocator,
    values: []const T,
    input_shape: []const usize,
    starts: []const usize,
    steps: []const usize,
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
        for (out_coords, input_coords, starts, steps) |coord, *input_coord, start, step| {
            input_coord.* = start + coord * step;
        }
        slot.* = values[common.coordsToLinear(input_coords, input_strides)];
    }
    return out;
}
