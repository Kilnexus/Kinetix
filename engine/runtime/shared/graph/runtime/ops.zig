const std = @import("std");
const onnx_metadata = @import("../onnx/metadata.zig");
const tensor_mod = @import("tensor.zig");

pub const Tensor = tensor_mod.Tensor;

pub fn isSupported(op_type: []const u8) bool {
    return std.mem.eql(u8, op_type, "Identity") or
        std.mem.eql(u8, op_type, "Add") or
        std.mem.eql(u8, op_type, "Mul") or
        std.mem.eql(u8, op_type, "Relu") or
        std.mem.eql(u8, op_type, "MatMul");
}

pub fn execute(
    allocator: std.mem.Allocator,
    node: onnx_metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    if (std.mem.eql(u8, node.op_type, "Identity")) return try identity(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Add")) return try elementwise(allocator, inputs, .add);
    if (std.mem.eql(u8, node.op_type, "Mul")) return try elementwise(allocator, inputs, .mul);
    if (std.mem.eql(u8, node.op_type, "Relu")) return try relu(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "MatMul")) return try matmul(allocator, inputs);
    return error.UnsupportedOnnxOperator;
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
