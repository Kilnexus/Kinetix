const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub const ReduceMode = enum {
    mean,
    sum,
    max,
};

pub fn softmax(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse -1, input.shape.len);
    const inner = common.elementCountFromShape(input.shape[axis + 1 ..]);
    const dim = input.shape[axis];
    const outer = common.elementCountFromShape(input.shape[0..axis]);
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (0..outer) |outer_index| {
        for (0..inner) |inner_index| {
            var max_value = -std.math.inf(f32);
            for (0..dim) |dim_index| {
                const index = outer_index * dim * inner + dim_index * inner + inner_index;
                max_value = @max(max_value, input.buffer.f32[index]);
            }
            var sum: f32 = 0;
            for (0..dim) |dim_index| {
                const index = outer_index * dim * inner + dim_index * inner + inner_index;
                const value = @exp(input.buffer.f32[index] - max_value);
                out[index] = value;
                sum += value;
            }
            if (sum == 0) return error.InvalidSoftmaxSum;
            for (0..dim) |dim_index| {
                const index = outer_index * dim * inner + dim_index * inner + inner_index;
                out[index] /= sum;
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn reduceMean(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    return try reduce(allocator, node, inputs, .mean);
}

pub fn reduceSum(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    return try reduce(allocator, node, inputs, .sum);
}

pub fn reduceMax(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    return try reduce(allocator, node, inputs, .max);
}

fn reduce(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor, mode: ReduceMode) !Tensor {
    if (inputs.len != 1 and inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const keepdims = (common.attributeInt(node, "keepdims") orelse 1) != 0;
    const raw_axes = if (inputs.len == 2)
        try common.indicesToOwnedI64(allocator, inputs[1].*)
    else
        try common.attributeIntsOwned(allocator, node, "axes");
    defer allocator.free(raw_axes);
    const axes = if (raw_axes.len == 0)
        try defaultAxesOwned(allocator, input.shape.len)
    else
        try common.normalizeAxesOwned(allocator, raw_axes, input.shape.len);
    defer allocator.free(axes);
    std.mem.sort(usize, axes, {}, comptime std.sort.asc(usize));
    try common.ensureUniqueAxes(axes);

    var out_shape_list = std.ArrayListUnmanaged(usize).empty;
    defer out_shape_list.deinit(allocator);
    for (input.shape, 0..) |dim, axis| {
        if (common.containsAxis(axes, axis)) {
            if (keepdims) try out_shape_list.append(allocator, 1);
        } else {
            try out_shape_list.append(allocator, dim);
        }
    }
    if (out_shape_list.items.len == 0) try out_shape_list.append(allocator, 1);

    const out_count = common.elementCountFromShape(out_shape_list.items);
    const out = try allocator.alloc(f32, out_count);
    errdefer allocator.free(out);
    @memset(out, initialReduceValue(mode));
    const counts = try allocator.alloc(usize, out_count);
    defer allocator.free(counts);
    @memset(counts, 0);

    const in_strides = try common.stridesOwned(allocator, input.shape);
    defer allocator.free(in_strides);
    const out_strides = try common.stridesOwned(allocator, out_shape_list.items);
    defer allocator.free(out_strides);
    const in_coords = try allocator.alloc(usize, input.shape.len);
    defer allocator.free(in_coords);
    const out_coords = try allocator.alloc(usize, out_shape_list.items.len);
    defer allocator.free(out_coords);

    for (input.buffer.f32, 0..) |value, linear| {
        common.linearToCoords(linear, input.shape, in_strides, in_coords);
        projectReduceCoords(in_coords, axes, keepdims, out_coords);
        const out_index = common.coordsToLinear(out_coords, out_strides);
        out[out_index] = switch (mode) {
            .mean, .sum => out[out_index] + value,
            .max => @max(out[out_index], value),
        };
        counts[out_index] += 1;
    }
    for (out, counts) |*value, count| {
        if (count == 0) return error.InvalidTensorShape;
        if (mode == .mean) value.* /= @floatFromInt(count);
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, out_shape_list.items),
        .buffer = .{ .f32 = out },
    };
}

fn initialReduceValue(mode: ReduceMode) f32 {
    return switch (mode) {
        .mean, .sum => 0,
        .max => -std.math.inf(f32),
    };
}

pub fn batchNormalization(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 5) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const scale = inputs[1].*;
    const bias = inputs[2].*;
    const mean = inputs[3].*;
    const variance = inputs[4].*;
    if (input.buffer != .f32 or scale.buffer != .f32 or bias.buffer != .f32 or mean.buffer != .f32 or variance.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len < 2) return error.UnsupportedTensorRank;
    const channels = input.shape[1];
    if (scale.elementCount() != channels or bias.elementCount() != channels or mean.elementCount() != channels or variance.elementCount() != channels) return error.ShapeMismatch;
    const epsilon = common.attributeFloat(node, "epsilon") orelse 0.00001;
    const spatial = common.elementCountFromShape(input.shape[2..]);
    const batch = input.shape[0];
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (0..batch) |n| {
        for (0..channels) |c| {
            const inv_std = 1.0 / @sqrt(variance.buffer.f32[c] + epsilon);
            const base = (n * channels + c) * spatial;
            for (0..spatial) |index| {
                const value = input.buffer.f32[base + index];
                out[base + index] = (value - mean.buffer.f32[c]) * inv_std * scale.buffer.f32[c] + bias.buffer.f32[c];
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

fn defaultAxesOwned(allocator: std.mem.Allocator, rank: usize) ![]usize {
    const axes = try allocator.alloc(usize, rank);
    errdefer allocator.free(axes);
    for (axes, 0..) |*axis, index| axis.* = index;
    return axes;
}

fn projectReduceCoords(input_coords: []const usize, axes: []const usize, keepdims: bool, out_coords: []usize) void {
    var out_index: usize = 0;
    for (input_coords, 0..) |coord, axis| {
        if (common.containsAxis(axes, axis)) {
            if (keepdims) {
                out_coords[out_index] = 0;
                out_index += 1;
            }
        } else {
            out_coords[out_index] = coord;
            out_index += 1;
        }
    }
    if (out_coords.len == 1 and out_index == 0) out_coords[0] = 0;
}

pub fn layerNormalization(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 2 or inputs.len > 3) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const scale = inputs[1].*;
    const bias: ?Tensor = if (inputs.len == 3) inputs[2].* else null;
    if (input.buffer != .f32 or scale.buffer != .f32) return error.UnsupportedTensorDType;
    if (bias) |bias_tensor| {
        if (bias_tensor.buffer != .f32) return error.UnsupportedTensorDType;
    }
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse -1, input.shape.len);
    const epsilon = common.attributeFloat(node, "epsilon") orelse 0.00001;
    const inner = common.elementCountFromShape(input.shape[axis..]);
    const outer = common.elementCountFromShape(input.shape[0..axis]);
    if (scale.elementCount() != inner) return error.ShapeMismatch;
    if (bias) |bias_tensor| {
        if (bias_tensor.elementCount() != inner) return error.ShapeMismatch;
    }
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (0..outer) |outer_index| {
        const base = outer_index * inner;
        var mean: f32 = 0;
        for (input.buffer.f32[base .. base + inner]) |value| mean += value;
        mean /= @floatFromInt(inner);
        var variance: f32 = 0;
        for (input.buffer.f32[base .. base + inner]) |value| {
            const centered = value - mean;
            variance += centered * centered;
        }
        variance /= @floatFromInt(inner);
        const inv_std = 1.0 / @sqrt(variance + epsilon);
        for (0..inner) |index| {
            const normalized = (input.buffer.f32[base + index] - mean) * inv_std;
            const bias_value = if (bias) |bias_tensor| bias_tensor.buffer.f32[index] else 0;
            out[base + index] = normalized * scale.buffer.f32[index] + bias_value;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn rmsNormalization(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    const scale = inputs[1].*;
    if (input.buffer != .f32 or scale.buffer != .f32) return error.UnsupportedTensorDType;
    const axis = try common.normalizeAxis(common.attributeInt(node, "axis") orelse -1, input.shape.len);
    const epsilon = common.attributeFloat(node, "epsilon") orelse 0.00001;
    const inner = common.elementCountFromShape(input.shape[axis..]);
    const outer = common.elementCountFromShape(input.shape[0..axis]);
    if (scale.elementCount() != inner) return error.ShapeMismatch;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (0..outer) |outer_index| {
        const base = outer_index * inner;
        var mean_square: f32 = 0;
        for (input.buffer.f32[base .. base + inner]) |value| mean_square += value * value;
        mean_square /= @floatFromInt(inner);
        const inv_rms = 1.0 / @sqrt(mean_square + epsilon);
        for (0..inner) |index| {
            out[base + index] = input.buffer.f32[base + index] * inv_rms * scale.buffer.f32[index];
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}
