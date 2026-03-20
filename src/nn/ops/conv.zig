const std = @import("std");
const types = @import("types.zig");
const common = @import("conv/common.zig");
const kernel_3x3 = @import("conv/kernel_3x3.zig");
const kernel_general = @import("conv/kernel_general.zig");
const kernel_pointwise = @import("conv/kernel_pointwise.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

pub fn conv2d(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
) OpError!void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = weights.shape[0];
    const kernel_in_channels = weights.shape[1];
    const kernel_h = weights.shape[2];
    const kernel_w = weights.shape[3];

    if (options.groups == 0 or in_channels % options.groups != 0 or out_channels % options.groups != 0) {
        return OpError.InvalidGroups;
    }
    if (kernel_in_channels != in_channels / options.groups) return OpError.ShapeMismatch;

    const expected_h = ((in_height + 2 * options.pad_h - kernel_h) / options.stride_h) + 1;
    const expected_w = ((in_width + 2 * options.pad_w - kernel_w) / options.stride_w) + 1;
    if (output.shape[0] != batch or output.shape[1] != out_channels or output.shape[2] != expected_h or output.shape[3] != expected_w) {
        return OpError.InvalidOutputShape;
    }
    if (bias) |bias_values| {
        if (bias_values.len != out_channels) return OpError.ShapeMismatch;
    }

    if (kernel_h == 1 and kernel_w == 1 and options.stride_h == 1 and options.stride_w == 1 and options.pad_h == 0 and options.pad_w == 0) {
        return kernel_pointwise.conv2dPointwise(input, weights, bias, output, options.groups, options.apply_silu);
    }
    if (kernel_h == 3 and kernel_w == 3 and options.pad_h == 1 and options.pad_w == 1 and options.groups == 1) {
        return kernel_3x3.conv2d3x3Pad1(input, weights, bias, output, options);
    }

    const workload = batch * out_channels * expected_h * expected_w * kernel_h * kernel_w * (in_channels / options.groups);
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return kernel_general.conv2dGeneralParallel(input, weights, bias, output, options, thread_count);
    }

    return kernel_general.conv2dGeneralRange(input, weights, bias, output, options, 0, out_channels);
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
