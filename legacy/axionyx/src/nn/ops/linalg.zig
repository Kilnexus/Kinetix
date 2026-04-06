const std = @import("std");
const types = @import("types.zig");

pub const OpError = types.OpError;

pub fn matmul(
    lhs: []const f32,
    rhs: []const f32,
    out: []f32,
    rows: usize,
    shared: usize,
    cols: usize,
) OpError!void {
    if (lhs.len != rows * shared or rhs.len != shared * cols or out.len != rows * cols) {
        return OpError.ShapeMismatch;
    }

    for (0..rows) |r| {
        for (0..cols) |c| {
            var acc: f32 = 0.0;
            for (0..shared) |k| {
                acc += lhs[r * shared + k] * rhs[k * cols + c];
            }
            out[r * cols + c] = acc;
        }
    }
}

pub fn softmaxRows(data: []f32, rows: usize, cols: usize) OpError!void {
    if (data.len != rows * cols) return OpError.ShapeMismatch;

    for (0..rows) |r| {
        const row = data[r * cols .. (r + 1) * cols];
        var max_value = row[0];
        for (row[1..]) |value| {
            if (value > max_value) max_value = value;
        }

        var sum: f32 = 0.0;
        for (row) |*value| {
            value.* = @exp(value.* - max_value);
            sum += value.*;
        }
        for (row) |*value| value.* /= sum;
    }
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
