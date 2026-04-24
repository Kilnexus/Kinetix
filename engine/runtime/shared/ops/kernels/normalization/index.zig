const std = @import("std");

pub const Error = error{
    ShapeMismatch,
};

pub fn rmsNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
) Error!void {
    if (output.len != input.len or input.len != weight.len) return error.ShapeMismatch;

    const inv_rms = computeInvRms(input, eps);
    applyRmsNormScaled(output, input, weight, inv_rms);
}

pub fn layerNorm(
    output: []f32,
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    eps: f32,
) Error!void {
    if (output.len != input.len or input.len != weight.len or weight.len != bias.len) return error.ShapeMismatch;
    if (input.len == 0) return error.ShapeMismatch;

    var mean: f32 = 0.0;
    for (input) |value| mean += value;
    mean /= @as(f32, @floatFromInt(input.len));

    var variance: f32 = 0.0;
    for (input) |value| {
        const centered = value - mean;
        variance += centered * centered;
    }
    variance /= @as(f32, @floatFromInt(input.len));

    const inv_std = 1.0 / @sqrt(variance + eps);
    for (output, input, weight, bias) |*out, x, w, b| {
        out.* = ((x - mean) * inv_std) * w + b;
    }
}

pub fn rmsNormRepeated(
    output: []f32,
    input: []const f32,
    repeat_count: usize,
    slice_len: usize,
    weight: []const f32,
    eps: f32,
) Error!void {
    if (weight.len != slice_len) return error.ShapeMismatch;
    if (input.len != repeat_count * slice_len) return error.ShapeMismatch;
    if (output.len != input.len) return error.ShapeMismatch;

    for (0..repeat_count) |idx| {
        const start = idx * slice_len;
        try rmsNorm(
            output[start .. start + slice_len],
            input[start .. start + slice_len],
            weight,
            eps,
        );
    }
}

fn computeInvRms(input: []const f32, eps: f32) f32 {
    var sum_vec0: @Vector(16, f32) = @splat(0.0);
    var sum_vec1: @Vector(16, f32) = @splat(0.0);
    var index: usize = 0;
    while (index + 32 <= input.len) : (index += 32) {
        const v0: @Vector(16, f32) = input[index..][0..16].*;
        const v1: @Vector(16, f32) = input[index + 16 ..][0..16].*;
        sum_vec0 += v0 * v0;
        sum_vec1 += v1 * v1;
    }

    var mean_square = @reduce(.Add, sum_vec0 + sum_vec1);
    while (index + 16 <= input.len) : (index += 16) {
        const v: @Vector(16, f32) = input[index..][0..16].*;
        mean_square += @reduce(.Add, v * v);
    }
    while (index < input.len) : (index += 1) {
        mean_square += input[index] * input[index];
    }

    mean_square /= @as(f32, @floatFromInt(input.len));
    return 1.0 / @sqrt(mean_square + eps);
}

fn applyRmsNormScaled(output: []f32, input: []const f32, weight: []const f32, inv_rms: f32) void {
    const inv_rms_vec: @Vector(16, f32) = @splat(inv_rms);
    var index: usize = 0;
    while (index + 16 <= output.len) : (index += 16) {
        const in_vec: @Vector(16, f32) = input[index..][0..16].*;
        const weight_vec: @Vector(16, f32) = weight[index..][0..16].*;
        output[index..][0..16].* = in_vec * weight_vec * inv_rms_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] = input[index] * inv_rms * weight[index];
    }
}

test "kernel normalization rmsNorm matches manual calculation" {
    const input = [_]f32{ 3.0, 4.0 };
    const weight = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0 };

    try rmsNorm(&output, &input, &weight, 0.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0.84852814), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.2627418), output[1], 1e-6);
}

test "kernel normalization layerNorm matches manual calculation" {
    const input = [_]f32{ 1.0, 2.0, 3.0 };
    const weight = [_]f32{ 1.0, 1.5, 0.5 };
    const bias = [_]f32{ 0.0, 0.25, -0.25 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try layerNorm(&output, &input, &weight, &bias, 0.0);

    try std.testing.expectApproxEqAbs(@as(f32, -1.2247448), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3623724), output[2], 1e-5);
}
