const std = @import("std");

pub fn siluValue(x: f32) f32 {
    return x / (1.0 + std.math.exp(-x));
}

pub fn sigmoidValue(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

pub fn geluValue(x: f32) f32 {
    const c = @sqrt(2.0 / std.math.pi);
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

pub fn siluInPlace(values: []f32) void {
    for (values) |*value| value.* = siluValue(value.*);
}

pub fn sigmoidInPlace(values: []f32) void {
    for (values) |*value| value.* = sigmoidValue(value.*);
}

pub fn add(output: []f32, lhs: []const f32, rhs: []const f32) !void {
    if (output.len != lhs.len or lhs.len != rhs.len) return error.SizeMismatch;
    for (output, lhs, rhs) |*out, left, right| out.* = left + right;
}

pub fn addInPlace(output: []f32, rhs: []const f32) !void {
    if (output.len != rhs.len) return error.SizeMismatch;
    addInPlaceUnchecked(output, rhs);
}

pub fn addInPlaceUnchecked(output: []f32, rhs: []const f32) void {
    const lane_count = 8;
    var index: usize = 0;
    while (index + lane_count <= output.len) : (index += lane_count) {
        const lhs_vec: @Vector(lane_count, f32) = output[index..][0..lane_count].*;
        const rhs_vec: @Vector(lane_count, f32) = rhs[index..][0..lane_count].*;
        output[index..][0..lane_count].* = lhs_vec + rhs_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] += rhs[index];
    }
}

pub fn geluInPlace(values: []f32) void {
    for (values) |*value| value.* = geluValue(value.*);
}

pub fn swiglu(output: []f32, gate: []const f32, up: []const f32) !void {
    if (output.len != gate.len or gate.len != up.len) return error.SizeMismatch;

    var index: usize = 0;
    while (index + 16 <= output.len) : (index += 16) {
        const gate_vec: @Vector(16, f32) = gate[index..][0..16].*;
        const up_vec: @Vector(16, f32) = up[index..][0..16].*;
        const silu_vec = gate_vec / (@as(@Vector(16, f32), @splat(1.0)) + @exp(-gate_vec));
        output[index..][0..16].* = silu_vec * up_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] = siluValue(gate[index]) * up[index];
    }
}

test "kernel activation geluInPlace matches known values" {
    var values = [_]f32{ -1.0, 0.0, 1.0 };
    geluInPlace(&values);

    try std.testing.expectApproxEqAbs(@as(f32, -0.158808), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), values[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841192), values[2], 1e-6);
}

test "kernel activation swiglu applies silu gate then multiplies up branch" {
    const gate = [_]f32{ 0.0, 1.0, -1.0 };
    const up = [_]f32{ 1.0, 2.0, 3.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try swiglu(&output, &gate, &up);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4621172), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8068243), output[2], 1e-6);
}

test "kernel activation add supports checked and unchecked paths" {
    const lhs = [_]f32{ 1.0, 2.0, 3.0 };
    const rhs = [_]f32{ 4.0, 5.0, 6.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0 };

    try add(&output, &lhs, &rhs);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5.0, 7.0, 9.0 }, &output);

    addInPlaceUnchecked(&output, &rhs);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 9.0, 12.0, 15.0 }, &output);
}
