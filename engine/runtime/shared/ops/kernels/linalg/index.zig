const std = @import("std");

pub const Error = error{
    ShapeMismatch,
};

pub fn dot(lhs: []const f32, rhs: []const f32) Error!f32 {
    if (lhs.len != rhs.len) return error.ShapeMismatch;

    var sum: f32 = 0.0;
    for (lhs, rhs) |a, b| sum += a * b;
    return sum;
}

pub fn axpyInPlace(output: []f32, alpha: f32, input: []const f32) Error!void {
    if (output.len != input.len) return error.ShapeMismatch;
    for (output, input) |*out, value| out.* += alpha * value;
}

pub fn matmulVec(
    output: []f32,
    weights_row_major: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) Error!void {
    if (output.len != rows) return error.ShapeMismatch;
    if (input.len != cols) return error.ShapeMismatch;
    if (weights_row_major.len != rows * cols) return error.ShapeMismatch;

    for (0..rows) |row| {
        const start = row * cols;
        output[row] = try dot(weights_row_major[start .. start + cols], input);
    }
}

pub fn matmul(
    lhs: []const f32,
    rhs: []const f32,
    out: []f32,
    rows: usize,
    shared: usize,
    cols: usize,
) Error!void {
    if (lhs.len != rows * shared or rhs.len != shared * cols or out.len != rows * cols) {
        return error.ShapeMismatch;
    }

    for (0..rows) |r| {
        for (0..cols) |c| {
            var acc: f32 = 0.0;
            for (0..shared) |k| acc += lhs[r * shared + k] * rhs[k * cols + c];
            out[r * cols + c] = acc;
        }
    }
}

pub fn softmaxRows(data: []f32, rows: usize, cols: usize) Error!void {
    if (data.len != rows * cols) return error.ShapeMismatch;

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

test "kernel linalg dot and matmulVec compute expected result" {
    const lhs = [_]f32{ 1.0, 2.0, 3.0 };
    const rhs = [_]f32{ 4.0, 5.0, 6.0 };
    try std.testing.expectEqual(@as(f32, 32.0), try dot(&lhs, &rhs));

    const weights = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const input = [_]f32{ 1.0, 0.5, -1.0 };
    var output = [_]f32{ 0.0, 0.0 };
    try matmulVec(&output, &weights, &input, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
}

test "kernel linalg softmax row sums to one" {
    var values = [_]f32{ 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
    try softmaxRows(&values, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[0] + values[1] + values[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[3] + values[4] + values[5], 1e-5);
}
