const std = @import("std");
const types = @import("types.zig");
const shared_kernels = @import("shared_ops").kernels;
const shared_conv = shared_kernels.conv;
const entry = shared_conv.entry;
const kernel_general = shared_conv.general;

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

fn toSharedOptions(options: Conv2DOptions) entry.Conv2DOptions {
    return .{
        .stride_h = options.stride_h,
        .stride_w = options.stride_w,
        .pad_h = options.pad_h,
        .pad_w = options.pad_w,
        .groups = options.groups,
        .apply_silu = options.apply_silu,
    };
}

fn mapSharedError(err: anyerror) OpError {
    return switch (err) {
        error.ShapeMismatch => OpError.ShapeMismatch,
        error.InvalidOutputShape => OpError.InvalidOutputShape,
        error.InvalidGroups => OpError.InvalidGroups,
        error.InvalidTensorRank => OpError.InvalidTensorRank,
        else => OpError.ShapeMismatch,
    };
}

pub fn conv2d(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
) OpError!void {
    return entry.conv2d(input, weights, bias, output, toSharedOptions(options)) catch |err| return mapSharedError(err);
}

pub fn conv2dPointwiseConcat(
    inputs: []const *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    apply_silu: bool,
) OpError!void {
    return entry.conv2dPointwiseConcat(inputs, weights, bias, output, apply_silu) catch |err| return mapSharedError(err);
}

test "conv2d 1x1 sums channels" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 2, 1, 1);
    defer input.deinit();
    input.data[0] = 2.0;
    input.data[1] = 3.0;

    var weights = try Tensor.init(testing.allocator, 1, 2, 1, 1);
    defer weights.deinit();
    weights.data[0] = 4.0;
    weights.data[1] = 5.0;

    var output = try Tensor.init(testing.allocator, 1, 1, 1, 1);
    defer output.deinit();

    try conv2d(&input, &weights, null, &output, .{});
    try testing.expectApproxEqAbs(@as(f32, 23.0), output.data[0], 1e-6);
}

test "depthwise conv2d works" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 2, 2, 2);
    defer input.deinit();
    @memcpy(input.data, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 });

    var weights = try Tensor.init(testing.allocator, 2, 1, 1, 1);
    defer weights.deinit();
    @memcpy(weights.data, &[_]f32{ 2, 3 });

    var output = try Tensor.init(testing.allocator, 1, 2, 2, 2);
    defer output.deinit();

    try conv2d(&input, &weights, null, &output, .{ .groups = 2 });
    try testing.expectApproxEqAbs(@as(f32, 2.0), output.get(0, 0, 0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 8.0), output.get(0, 0, 1, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 15.0), output.get(0, 1, 0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 24.0), output.get(0, 1, 1, 1), 1e-6);
}

test "conv2d 3x3 stride2 fast path matches general path" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 2, 5, 5);
    defer input.deinit();
    for (input.data, 0..) |*value, index| value.* = @as(f32, @floatFromInt(index + 1)) * 0.125;

    var weights = try Tensor.init(testing.allocator, 3, 2, 3, 3);
    defer weights.deinit();
    for (weights.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 11) + 1)) - 6.0) * 0.15;
    }

    const bias_values = [_]f32{ 0.25, -0.5, 1.0 };

    var fast = try Tensor.init(testing.allocator, 1, 3, 3, 3);
    defer fast.deinit();
    var general = try Tensor.init(testing.allocator, 1, 3, 3, 3);
    defer general.deinit();

    try conv2d(&input, &weights, &bias_values, &fast, .{
        .stride_h = 2,
        .stride_w = 2,
        .pad_h = 1,
        .pad_w = 1,
    });
    try kernel_general.conv2dGeneralRange(&input, &weights, &bias_values, &general, .{
        .stride_h = 2,
        .stride_w = 2,
        .pad_h = 1,
        .pad_w = 1,
    }, 0, 3);

    for (fast.data, general.data) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "conv2d 3x3 stride1 fast path matches general path on wide tensor" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 5, 19, 27);
    defer input.deinit();
    for (input.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 23) + 1)) - 12.0) * 0.09;
    }

    var weights = try Tensor.init(testing.allocator, 7, 5, 3, 3);
    defer weights.deinit();
    for (weights.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 17) + 1)) - 9.0) * 0.06;
    }

    const bias_values = [_]f32{ -0.2, 0.15, 0.0, 0.4, -0.35, 0.05, 0.3 };

    var fast = try Tensor.init(testing.allocator, 1, 7, 19, 27);
    defer fast.deinit();
    var general = try Tensor.init(testing.allocator, 1, 7, 19, 27);
    defer general.deinit();

    try conv2d(&input, &weights, &bias_values, &fast, .{
        .pad_h = 1,
        .pad_w = 1,
    });
    try kernel_general.conv2dGeneralRange(&input, &weights, &bias_values, &general, .{
        .pad_h = 1,
        .pad_w = 1,
    }, 0, 7);

    for (fast.data, general.data) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "conv2d 3x3 stride2 fast path matches general path on wide tensor" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 3, 33, 35);
    defer input.deinit();
    for (input.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 29) + 1)) - 15.0) * 0.08;
    }

    var weights = try Tensor.init(testing.allocator, 11, 3, 3, 3);
    defer weights.deinit();
    for (weights.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 31) + 1)) - 16.0) * 0.05;
    }

    const bias_values = [_]f32{ 0.1, -0.2, 0.35, -0.15, 0.0, 0.25, -0.05, 0.3, -0.4, 0.2, 0.45 };

    var fast = try Tensor.init(testing.allocator, 1, 11, 17, 18);
    defer fast.deinit();
    var general = try Tensor.init(testing.allocator, 1, 11, 17, 18);
    defer general.deinit();

    try conv2d(&input, &weights, &bias_values, &fast, .{
        .stride_h = 2,
        .stride_w = 2,
        .pad_h = 1,
        .pad_w = 1,
    });
    try kernel_general.conv2dGeneralRange(&input, &weights, &bias_values, &general, .{
        .stride_h = 2,
        .stride_w = 2,
        .pad_h = 1,
        .pad_w = 1,
    }, 0, 11);

    for (fast.data, general.data) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "pointwise concat fast path matches materialized concat" {
    const testing = std.testing;

    var lhs = try Tensor.init(testing.allocator, 1, 2, 2, 2);
    defer lhs.deinit();
    for (lhs.data, 0..) |*value, index| value.* = @as(f32, @floatFromInt(index + 1)) * 0.25;

    var rhs = try Tensor.init(testing.allocator, 1, 3, 2, 2);
    defer rhs.deinit();
    for (rhs.data, 0..) |*value, index| value.* = (@as(f32, @floatFromInt(index + 2)) - 3.0) * 0.2;

    var weights = try Tensor.init(testing.allocator, 4, 5, 1, 1);
    defer weights.deinit();
    for (weights.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 7) + 1)) - 4.0) * 0.1;
    }

    const bias_values = [_]f32{ 0.1, -0.2, 0.3, -0.4 };

    var materialized = try Tensor.init(testing.allocator, 1, 5, 2, 2);
    defer materialized.deinit();
    const inputs = [_]*const Tensor{ &lhs, &rhs };
    try @import("layout.zig").concatChannels(&inputs, &materialized);

    var expected = try Tensor.init(testing.allocator, 1, 4, 2, 2);
    defer expected.deinit();
    try conv2d(&materialized, &weights, &bias_values, &expected, .{});

    var actual = try Tensor.init(testing.allocator, 1, 4, 2, 2);
    defer actual.deinit();
    try conv2dPointwiseConcat(&inputs, &weights, &bias_values, &actual, false);

    for (actual.data, expected.data) |lhs_value, rhs_value| {
        try testing.expectApproxEqAbs(rhs_value, lhs_value, 1e-6);
    }
}

test "pointwise concat fast path matches materialized concat with three inputs" {
    const testing = std.testing;

    var a = try Tensor.init(testing.allocator, 1, 2, 3, 3);
    defer a.deinit();
    for (a.data, 0..) |*value, index| value.* = (@as(f32, @floatFromInt(index + 1)) - 5.0) * 0.2;

    var b = try Tensor.init(testing.allocator, 1, 3, 3, 3);
    defer b.deinit();
    for (b.data, 0..) |*value, index| value.* = (@as(f32, @floatFromInt(index % 13)) - 6.0) * 0.17;

    var c = try Tensor.init(testing.allocator, 1, 4, 3, 3);
    defer c.deinit();
    for (c.data, 0..) |*value, index| value.* = (@as(f32, @floatFromInt((index * 3) % 17)) - 8.0) * 0.11;

    var weights = try Tensor.init(testing.allocator, 5, 9, 1, 1);
    defer weights.deinit();
    for (weights.data, 0..) |*value, index| {
        value.* = (@as(f32, @floatFromInt((index % 19) + 1)) - 10.0) * 0.07;
    }

    const bias_values = [_]f32{ -0.3, 0.1, 0.25, -0.05, 0.4 };

    var materialized = try Tensor.init(testing.allocator, 1, 9, 3, 3);
    defer materialized.deinit();
    const inputs = [_]*const Tensor{ &a, &b, &c };
    try @import("layout.zig").concatChannels(&inputs, &materialized);

    var expected = try Tensor.init(testing.allocator, 1, 5, 3, 3);
    defer expected.deinit();
    try conv2d(&materialized, &weights, &bias_values, &expected, .{});

    var actual = try Tensor.init(testing.allocator, 1, 5, 3, 3);
    defer actual.deinit();
    try conv2dPointwiseConcat(&inputs, &weights, &bias_values, &actual, false);

    for (actual.data, expected.data) |lhs_value, rhs_value| {
        try testing.expectApproxEqAbs(rhs_value, lhs_value, 1e-6);
    }
}
