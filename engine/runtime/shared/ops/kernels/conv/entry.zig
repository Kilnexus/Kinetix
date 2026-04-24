const std = @import("std");
const common = @import("common.zig");
const kernel_3x3 = @import("kernel_3x3.zig");
const kernel_general = @import("general.zig");
const kernel_pointwise = @import("pointwise.zig");
const env = @import("engine_env");

pub const Tensor = common.Tensor;
pub const OpError = common.OpError;
pub const Conv2DOptions = common.Conv2DOptions;

fn envFlagEnabled(name: []const u8) bool {
    const value = env.getOwned(std.heap.page_allocator, name) catch return false;
    defer std.heap.page_allocator.free(value);
    return value.len != 0 and !std.mem.eql(u8, value, "0");
}

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
        return error.InvalidGroups;
    }
    if (kernel_in_channels != in_channels / options.groups) return error.ShapeMismatch;

    const expected_h = ((in_height + 2 * options.pad_h - kernel_h) / options.stride_h) + 1;
    const expected_w = ((in_width + 2 * options.pad_w - kernel_w) / options.stride_w) + 1;
    if (output.shape[0] != batch or output.shape[1] != out_channels or output.shape[2] != expected_h or output.shape[3] != expected_w) {
        return error.InvalidOutputShape;
    }
    if (bias) |bias_values| {
        if (bias_values.len != out_channels) return error.ShapeMismatch;
    }

    const disable_pointwise_fastpath = envFlagEnabled("KINETIX_CONV_DISABLE_POINTWISE_FASTPATH");
    const disable_3x3_fastpath = envFlagEnabled("KINETIX_CONV_DISABLE_3X3_FASTPATH");

    if (!disable_pointwise_fastpath and kernel_h == 1 and kernel_w == 1 and options.stride_h == 1 and options.stride_w == 1 and options.pad_h == 0 and options.pad_w == 0) {
        return kernel_pointwise.conv2dPointwise(input, weights, bias, output, options.groups, options.apply_silu);
    }
    if (!disable_3x3_fastpath and kernel_h == 3 and kernel_w == 3 and options.pad_h == 1 and options.pad_w == 1 and options.groups == 1) {
        return kernel_3x3.conv2d3x3Pad1(input, weights, bias, output, options);
    }

    const workload = batch * out_channels * expected_h * expected_w * kernel_h * kernel_w * (in_channels / options.groups);
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return kernel_general.conv2dGeneralParallel(input, weights, bias, output, options, thread_count);
    }

    return kernel_general.conv2dGeneralRange(input, weights, bias, output, options, 0, out_channels);
}

pub fn conv2dPointwiseConcat(
    inputs: []const *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    apply_silu: bool,
) OpError!void {
    return kernel_pointwise.conv2dPointwiseConcat(inputs, weights, bias, output, apply_silu);
}
