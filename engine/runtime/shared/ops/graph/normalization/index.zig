const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

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
