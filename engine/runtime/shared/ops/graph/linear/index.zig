const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub fn matmul(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
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

pub fn gemm(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 2 or inputs.len > 3) return error.InvalidOperatorArity;
    const a = inputs[0].*;
    const b = inputs[1].*;
    if (a.buffer != .f32 or b.buffer != .f32) return error.UnsupportedTensorDType;
    if (a.shape.len != 2 or b.shape.len != 2) return error.UnsupportedTensorRank;
    const trans_a = common.attributeInt(node, "transA") orelse 0;
    const trans_b = common.attributeInt(node, "transB") orelse 0;
    const alpha = common.attributeFloat(node, "alpha") orelse 1;
    const beta = common.attributeFloat(node, "beta") orelse 1;
    const m = if (trans_a != 0) a.shape[1] else a.shape[0];
    const k = if (trans_a != 0) a.shape[0] else a.shape[1];
    const b_k = if (trans_b != 0) b.shape[1] else b.shape[0];
    const n = if (trans_b != 0) b.shape[0] else b.shape[1];
    if (k != b_k) return error.ShapeMismatch;

    const out = try allocator.alloc(f32, m * n);
    errdefer allocator.free(out);
    const c: ?Tensor = if (inputs.len == 3) inputs[2].* else null;
    if (c) |bias| {
        if (bias.buffer != .f32) return error.UnsupportedTensorDType;
    }
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |inner_index| {
                const a_value = if (trans_a != 0)
                    a.buffer.f32[inner_index * a.shape[1] + row]
                else
                    a.buffer.f32[row * a.shape[1] + inner_index];
                const b_value = if (trans_b != 0)
                    b.buffer.f32[col * b.shape[1] + inner_index]
                else
                    b.buffer.f32[inner_index * b.shape[1] + col];
                sum += a_value * b_value;
            }
            const bias = if (c) |bias_tensor| try gemmBiasValue(bias_tensor, row, col, m, n) else 0;
            out[row * n + col] = alpha * sum + beta * bias;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ m, n }),
        .buffer = .{ .f32 = out },
    };
}

fn gemmBiasValue(bias: Tensor, row: usize, col: usize, m: usize, n: usize) !f32 {
    if (bias.elementCount() == 1) return bias.buffer.f32[0];
    if (bias.shape.len == 1) {
        if (bias.shape[0] == n) return bias.buffer.f32[col];
        if (bias.shape[0] == m) return bias.buffer.f32[row];
        return error.ShapeMismatch;
    }
    if (bias.shape.len == 2) {
        if (bias.shape[0] == m and bias.shape[1] == n) return bias.buffer.f32[row * n + col];
        if (bias.shape[0] == 1 and bias.shape[1] == n) return bias.buffer.f32[col];
        if (bias.shape[0] == m and bias.shape[1] == 1) return bias.buffer.f32[row];
    }
    return error.ShapeMismatch;
}
