const std = @import("std");
const kernels = @import("shared_ops").kernels;

pub fn dot(lhs: []const f32, rhs: []const f32) !f32 {
    return kernels.linalg.dot(lhs, rhs) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

pub fn axpyInPlace(output: []f32, alpha: f32, input: []const f32) !void {
    return kernels.linalg.axpyInPlace(output, alpha, input) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

pub fn matmulVec(
    output: []f32,
    weights_row_major: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) !void {
    return kernels.linalg.matmulVec(output, weights_row_major, input, rows, cols) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

pub fn rmsNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) !void {
    return kernels.normalization.rmsNorm(output, input, weight, eps) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

pub fn layerNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) !void {
    return kernels.normalization.layerNorm(output, input, weight, bias, eps) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

pub fn rmsNormRepeated(
    output: []f32,
    input: []const f32,
    repeat_count: usize,
    slice_len: usize,
    weight: []const f32,
    eps: f32,
) !void {
    return kernels.normalization.rmsNormRepeated(output, input, repeat_count, slice_len, weight, eps) catch |err| switch (err) {
        error.ShapeMismatch => return error.SizeMismatch,
    };
}

fn rmsNormScalarReference(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) void {
    var mean_square: f32 = 0.0;
    for (input) |value| {
        mean_square += value * value;
    }
    mean_square /= @as(f32, @floatFromInt(input.len));

    const inv_rms = 1.0 / @sqrt(mean_square + eps);
    for (output, input, weight) |*out, x, w| {
        out.* = x * inv_rms * w;
    }
}

pub fn silu(x: f32) f32 {
    return kernels.activation.siluValue(x);
}

pub fn gelu(x: f32) f32 {
    return kernels.activation.geluValue(x);
}

pub fn geluInPlace(values: []f32) void {
    kernels.activation.geluInPlace(values);
}

pub fn swiglu(output: []f32, gate: []const f32, up: []const f32) !void {
    return kernels.activation.swiglu(output, gate, up);
}

fn swigluScalarReference(output: []f32, gate: []const f32, up: []const f32) void {
    for (output, gate, up) |*out, gate_value, up_value| {
        out.* = silu(gate_value) * up_value;
    }
}

test "dot computes expected result" {
    const testing = std.testing;

    const lhs = [_]f32{ 1.0, 2.0, 3.0 };
    const rhs = [_]f32{ 4.0, 5.0, 6.0 };
    try testing.expectEqual(@as(f32, 32.0), try dot(&lhs, &rhs));
}

test "axpyInPlace accumulates scaled input into output" {
    const testing = std.testing;

    var output = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = [_]f32{ 0.5, -1.0, 2.0, 1.5 };

    try axpyInPlace(&output, 2.0, &input);

    try testing.expectApproxEqAbs(@as(f32, 2.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0), output[3], 1e-6);
}

test "matmulVec multiplies row-major matrix by vector" {
    const testing = std.testing;

    const weights = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const input = [_]f32{ 1.0, 0.5, -1.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try matmulVec(&output, &weights, &input, 2, 3);

    try testing.expectApproxEqAbs(@as(f32, -1.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 1e-6);
}

test "rmsNorm matches manual calculation" {
    const testing = std.testing;

    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try rmsNorm(&output, &input, &weight, 0.0);

    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[1], 1e-6);
}

test "wide rmsNorm matches scalar reference" {
    const testing = std.testing;

    inline for (.{ 128, 1024, 3072 }) |len| {
        const input = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(input);
        const weight = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(weight);
        const output = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(output);
        const expected = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(expected);

        for (input, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 17 + 3) % 41)) - 20)) / 8.0;
            weight[idx] = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 11 + 7) % 37)) - 18)) / 9.0;
        }

        rmsNormScalarReference(expected, input, weight, 1e-5);
        try rmsNorm(output, input, weight, 1e-5);

        for (expected, output) |want, got| {
            try testing.expectApproxEqAbs(want, got, 1e-5);
        }
    }
}

test "layerNorm matches manual calculation" {
    const testing = std.testing;

    const input = [_]f32{ 1.0, 2.0, 3.0 };
    const weight = [_]f32{ 1.0, 1.5, 0.5 };
    const bias = [_]f32{ 0.0, 0.25, -0.25 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try layerNorm(&output, &input, &weight, &bias, 0.0);

    try testing.expectApproxEqAbs(@as(f32, -1.2247448), output[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.25), output[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.3623724), output[2], 1e-5);
}

test "swiglu applies silu gate then multiplies up branch" {
    const testing = std.testing;

    const gate = [_]f32{ 0.0, 1.0, -1.0 };
    const up = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try swiglu(&output, &gate, &up);

    try testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.4621172), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.8068243), output[2], 1e-6);
}

test "wide swiglu matches scalar reference" {
    const testing = std.testing;

    inline for (.{ 128, 1024, 3072 }) |len| {
        const gate = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(gate);
        const up = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(up);
        const output = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(output);
        const expected = try testing.allocator.alloc(f32, len);
        defer testing.allocator.free(expected);

        for (gate, 0..) |*value, idx| {
            value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 13 + 5) % 43)) - 21)) / 8.0;
            up[idx] = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 9 + 1) % 39)) - 19)) / 7.0;
        }

        swigluScalarReference(expected, gate, up);
        try swiglu(output, gate, up);

        for (expected, output) |want, got| {
            try testing.expectApproxEqAbs(want, got, 1e-6);
        }
    }
}

test "geluInPlace matches known values" {
    const testing = std.testing;

    var values = [_]f32{ -1.0, 0.0, 1.0 };
    geluInPlace(&values);

    try testing.expectApproxEqAbs(@as(f32, -0.158808), values[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), values[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.841192), values[2], 1e-6);
}

test "rmsNormRepeated applies same norm weight to multiple slices" {
    const testing = std.testing;

    const input = [_]f32{ 3.0, 4.0, 6.0, 8.0 };
    const weight = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try rmsNormRepeated(&output, &input, 2, 2, &weight, 0.0);

    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.84852814), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.2627418), output[3], 1e-6);
}
