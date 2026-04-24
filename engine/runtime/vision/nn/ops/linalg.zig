const std = @import("std");
const types = @import("types.zig");
const kernels = @import("shared_ops").kernels;

pub const OpError = types.OpError;

pub fn matmul(
    lhs: []const f32,
    rhs: []const f32,
    out: []f32,
    rows: usize,
    shared: usize,
    cols: usize,
) OpError!void {
    kernels.linalg.matmul(lhs, rhs, out, rows, shared, cols) catch |err| switch (err) {
        error.ShapeMismatch => return OpError.ShapeMismatch,
    };
}

pub fn softmaxRows(data: []f32, rows: usize, cols: usize) OpError!void {
    kernels.linalg.softmaxRows(data, rows, cols) catch |err| switch (err) {
        error.ShapeMismatch => return OpError.ShapeMismatch,
    };
}

test "softmax row sums to one" {
    const testing = std.testing;
    var values = [_]f32{ 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
    try softmaxRows(&values, 2, 3);

    try testing.expectApproxEqAbs(@as(f32, 1.0), values[0] + values[1] + values[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), values[3] + values[4] + values[5], 1e-5);
    try testing.expect(values[2] > values[1]);
    try testing.expect(values[1] > values[0]);
}
