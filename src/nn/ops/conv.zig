const std = @import("std");
const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

const max_supported_conv_threads = 4;
const conv_parallel_min_workload = 2_000_000;

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
        return conv2dPointwise(input, weights, bias, output, options.groups);
    }
    if (kernel_h == 3 and kernel_w == 3 and options.pad_h == 1 and options.pad_w == 1 and options.groups == 1) {
        return conv2d3x3Pad1(input, weights, bias, output, options);
    }

    const workload = batch * out_channels * expected_h * expected_w * kernel_h * kernel_w * (in_channels / options.groups);
    const thread_count = chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2dGeneralParallel(input, weights, bias, output, options, thread_count);
    }

    return conv2dGeneralRange(input, weights, bias, output, options, 0, out_channels);
}

fn conv2d3x3Pad1(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
) OpError!void {
    if (options.stride_h == 2 and options.stride_w == 2) {
        return conv2d3x3Pad1Stride2(input, weights, bias, output, options);
    }

    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
    const thread_count = chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2d3x3Pad1Parallel(input, weights, bias, output, options, thread_count);
    }

    return conv2d3x3Pad1Range(input, weights, bias, output, options, 0, out_channels);
}

fn conv2d3x3Pad1Stride2(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
) OpError!void {
    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
    const thread_count = chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2d3x3Pad1Stride2Parallel(input, weights, bias, output, options, thread_count);
    }

    return conv2d3x3Pad1Stride2Range(input, weights, bias, output, 0, out_channels);
}

fn conv2d3x3Pad1Parallel(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    thread_count: usize,
) OpError!void {
    var threads: [max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2d3x3Pad1Range(input, weights, bias, output, options, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2d3x3Pad1Worker, .{
                Conv2DTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .options = options,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                try conv2d3x3Pad1Range(input, weights, bias, output, options, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

fn conv2d3x3Pad1Stride2Parallel(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    thread_count: usize,
) OpError!void {
    var threads: [max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2d3x3Pad1Stride2Range(input, weights, bias, output, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2d3x3Pad1Stride2Worker, .{
                Conv2DTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .options = options,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                try conv2d3x3Pad1Stride2Range(input, weights, bias, output, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

fn conv2d3x3Pad1Range(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
) OpError!void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const input_plane = in_height * in_width;
    const output_plane = expected_h * expected_w;
    const stride_h: isize = @intCast(options.stride_h);
    const stride_w: isize = @intCast(options.stride_w);

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * input_plane;
        const output_batch_base = n * out_channels * output_plane;
        var oc = oc_start;
        while (oc + 1 < oc_end) : (oc += 2) {
            const weights0_channel_base = oc * in_channels * 9;
            const weights1_channel_base = (oc + 1) * in_channels * 9;
            const output0_channel_base = output_batch_base + oc * output_plane;
            const output1_channel_base = output_batch_base + (oc + 1) * output_plane;
            const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;

            for (0..expected_h) |oy| {
                const base_y = @as(isize, @intCast(oy)) * stride_h - 1;
                const output0_row_base = output0_channel_base + oy * expected_w;
                const output1_row_base = output1_channel_base + oy * expected_w;
                for (0..expected_w) |ox| {
                    const base_x = @as(isize, @intCast(ox)) * stride_w - 1;
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;

                    const interior =
                        base_y >= 0 and base_x >= 0 and
                        base_y + 2 < @as(isize, @intCast(in_height)) and
                        base_x + 2 < @as(isize, @intCast(in_width));

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
                        const weight0_base = weights0_channel_base + ic * 9;
                        const weight1_base = weights1_channel_base + ic * 9;
                        if (interior) {
                            const row0 = @as(usize, @intCast(base_y)) * in_width + @as(usize, @intCast(base_x));
                            const row1 = row0 + in_width;
                            const row2 = row1 + in_width;
                            const v00 = input_channel[row0];
                            const v01 = input_channel[row0 + 1];
                            const v02 = input_channel[row0 + 2];
                            const v10 = input_channel[row1];
                            const v11 = input_channel[row1 + 1];
                            const v12 = input_channel[row1 + 2];
                            const v20 = input_channel[row2];
                            const v21 = input_channel[row2 + 1];
                            const v22 = input_channel[row2 + 2];

                            acc0 += v00 * weights.data[weight0_base];
                            acc0 += v01 * weights.data[weight0_base + 1];
                            acc0 += v02 * weights.data[weight0_base + 2];
                            acc0 += v10 * weights.data[weight0_base + 3];
                            acc0 += v11 * weights.data[weight0_base + 4];
                            acc0 += v12 * weights.data[weight0_base + 5];
                            acc0 += v20 * weights.data[weight0_base + 6];
                            acc0 += v21 * weights.data[weight0_base + 7];
                            acc0 += v22 * weights.data[weight0_base + 8];

                            acc1 += v00 * weights.data[weight1_base];
                            acc1 += v01 * weights.data[weight1_base + 1];
                            acc1 += v02 * weights.data[weight1_base + 2];
                            acc1 += v10 * weights.data[weight1_base + 3];
                            acc1 += v11 * weights.data[weight1_base + 4];
                            acc1 += v12 * weights.data[weight1_base + 5];
                            acc1 += v20 * weights.data[weight1_base + 6];
                            acc1 += v21 * weights.data[weight1_base + 7];
                            acc1 += v22 * weights.data[weight1_base + 8];
                        } else {
                            const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
                            const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
                            const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
                            const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

                            var iy = y_start;
                            while (iy < y_end) : (iy += 1) {
                                const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
                                const input_row_base = iy * in_width;
                                const weight0_row_base = weight0_base + ky * 3;
                                const weight1_row_base = weight1_base + ky * 3;
                                var ix = x_start;
                                while (ix < x_end) : (ix += 1) {
                                    const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                                    const v = input_channel[input_row_base + ix];
                                    acc0 += v * weights.data[weight0_row_base + kx];
                                    acc1 += v * weights.data[weight1_row_base + kx];
                                }
                            }
                        }
                    }

                    output.data[output0_row_base + ox] = acc0;
                    output.data[output1_row_base + ox] = acc1;
                }
            }
        }

        while (oc < oc_end) : (oc += 1) {
            const weights_channel_base = oc * in_channels * 9;
            const output_channel_base = output_batch_base + oc * output_plane;
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;

            for (0..expected_h) |oy| {
                const base_y = @as(isize, @intCast(oy)) * stride_h - 1;
                const output_row_base = output_channel_base + oy * expected_w;
                for (0..expected_w) |ox| {
                    const base_x = @as(isize, @intCast(ox)) * stride_w - 1;
                    var acc: f32 = bias_value;

                    const interior =
                        base_y >= 0 and base_x >= 0 and
                        base_y + 2 < @as(isize, @intCast(in_height)) and
                        base_x + 2 < @as(isize, @intCast(in_width));

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
                        const weight_base = weights_channel_base + ic * 9;
                        if (interior) {
                            const row0 = @as(usize, @intCast(base_y)) * in_width + @as(usize, @intCast(base_x));
                            const row1 = row0 + in_width;
                            const row2 = row1 + in_width;
                            acc += input_channel[row0] * weights.data[weight_base];
                            acc += input_channel[row0 + 1] * weights.data[weight_base + 1];
                            acc += input_channel[row0 + 2] * weights.data[weight_base + 2];
                            acc += input_channel[row1] * weights.data[weight_base + 3];
                            acc += input_channel[row1 + 1] * weights.data[weight_base + 4];
                            acc += input_channel[row1 + 2] * weights.data[weight_base + 5];
                            acc += input_channel[row2] * weights.data[weight_base + 6];
                            acc += input_channel[row2 + 1] * weights.data[weight_base + 7];
                            acc += input_channel[row2 + 2] * weights.data[weight_base + 8];
                        } else {
                            const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
                            const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
                            const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
                            const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

                            var iy = y_start;
                            while (iy < y_end) : (iy += 1) {
                                const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
                                const input_row_base = iy * in_width;
                                const weight_row_base = weight_base + ky * 3;
                                var ix = x_start;
                                while (ix < x_end) : (ix += 1) {
                                    const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                                    acc += input_channel[input_row_base + ix] * weights.data[weight_row_base + kx];
                                }
                            }
                        }
                    }

                    output.data[output_row_base + ox] = acc;
                }
            }
        }
    }
}

fn conv2d3x3Pad1Stride2Range(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    oc_start: usize,
    oc_end: usize,
) OpError!void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const input_plane = in_height * in_width;
    const output_plane = expected_h * expected_w;
    const interior_h_end = @min(expected_h, in_height / 2);
    const interior_w_end = @min(expected_w, in_width / 2);

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * input_plane;
        const output_batch_base = n * out_channels * output_plane;
        var oc = oc_start;
        while (oc + 1 < oc_end) : (oc += 2) {
            const weights0_channel_base = oc * in_channels * 9;
            const weights1_channel_base = (oc + 1) * in_channels * 9;
            const output0_channel_base = output_batch_base + oc * output_plane;
            const output1_channel_base = output_batch_base + (oc + 1) * output_plane;
            const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;

            for (0..expected_h) |oy| {
                const output0_row_base = output0_channel_base + oy * expected_w;
                const output1_row_base = output1_channel_base + oy * expected_w;
                if (oy == 0 or oy >= interior_h_end) {
                    for (0..expected_w) |ox| {
                        output.data[output0_row_base + ox] = conv2d3x3Pad1Stride2Point(
                            input,
                            weights,
                            bias0,
                            input_batch_base,
                            weights0_channel_base,
                            oy,
                            ox,
                        );
                        output.data[output1_row_base + ox] = conv2d3x3Pad1Stride2Point(
                            input,
                            weights,
                            bias1,
                            input_batch_base,
                            weights1_channel_base,
                            oy,
                            ox,
                        );
                    }
                    continue;
                }

                output.data[output0_row_base] = conv2d3x3Pad1Stride2Point(
                    input,
                    weights,
                    bias0,
                    input_batch_base,
                    weights0_channel_base,
                    oy,
                    0,
                );
                output.data[output1_row_base] = conv2d3x3Pad1Stride2Point(
                    input,
                    weights,
                    bias1,
                    input_batch_base,
                    weights1_channel_base,
                    oy,
                    0,
                );

                for (1..interior_w_end) |ox| {
                    const output0_index = output0_row_base + ox;
                    const output1_index = output1_row_base + ox;
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    const row0 = (oy * 2 - 1) * in_width + (ox * 2 - 1);
                    const row1 = row0 + in_width;
                    const row2 = row1 + in_width;

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
                        const weight0_base = weights0_channel_base + ic * 9;
                        const weight1_base = weights1_channel_base + ic * 9;
                        const v00 = input_channel[row0];
                        const v01 = input_channel[row0 + 1];
                        const v02 = input_channel[row0 + 2];
                        const v10 = input_channel[row1];
                        const v11 = input_channel[row1 + 1];
                        const v12 = input_channel[row1 + 2];
                        const v20 = input_channel[row2];
                        const v21 = input_channel[row2 + 1];
                        const v22 = input_channel[row2 + 2];

                        acc0 += v00 * weights.data[weight0_base];
                        acc0 += v01 * weights.data[weight0_base + 1];
                        acc0 += v02 * weights.data[weight0_base + 2];
                        acc0 += v10 * weights.data[weight0_base + 3];
                        acc0 += v11 * weights.data[weight0_base + 4];
                        acc0 += v12 * weights.data[weight0_base + 5];
                        acc0 += v20 * weights.data[weight0_base + 6];
                        acc0 += v21 * weights.data[weight0_base + 7];
                        acc0 += v22 * weights.data[weight0_base + 8];

                        acc1 += v00 * weights.data[weight1_base];
                        acc1 += v01 * weights.data[weight1_base + 1];
                        acc1 += v02 * weights.data[weight1_base + 2];
                        acc1 += v10 * weights.data[weight1_base + 3];
                        acc1 += v11 * weights.data[weight1_base + 4];
                        acc1 += v12 * weights.data[weight1_base + 5];
                        acc1 += v20 * weights.data[weight1_base + 6];
                        acc1 += v21 * weights.data[weight1_base + 7];
                        acc1 += v22 * weights.data[weight1_base + 8];
                    }

                    output.data[output0_index] = acc0;
                    output.data[output1_index] = acc1;
                }

                for (interior_w_end..expected_w) |ox| {
                    output.data[output0_row_base + ox] = conv2d3x3Pad1Stride2Point(
                        input,
                        weights,
                        bias0,
                        input_batch_base,
                        weights0_channel_base,
                        oy,
                        ox,
                    );
                    output.data[output1_row_base + ox] = conv2d3x3Pad1Stride2Point(
                        input,
                        weights,
                        bias1,
                        input_batch_base,
                        weights1_channel_base,
                        oy,
                        ox,
                    );
                }
            }
        }

        while (oc < oc_end) : (oc += 1) {
            const weights_channel_base = oc * in_channels * 9;
            const output_channel_base = output_batch_base + oc * output_plane;
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;

            for (0..expected_h) |oy| {
                const output_row_base = output_channel_base + oy * expected_w;
                if (oy == 0 or oy >= interior_h_end) {
                    for (0..expected_w) |ox| {
                        output.data[output_row_base + ox] = conv2d3x3Pad1Stride2Point(
                            input,
                            weights,
                            bias_value,
                            input_batch_base,
                            weights_channel_base,
                            oy,
                            ox,
                        );
                    }
                    continue;
                }

                output.data[output_row_base] = conv2d3x3Pad1Stride2Point(
                    input,
                    weights,
                    bias_value,
                    input_batch_base,
                    weights_channel_base,
                    oy,
                    0,
                );

                for (1..interior_w_end) |ox| {
                    const output_index = output_row_base + ox;
                    var acc: f32 = bias_value;
                    const row0 = (oy * 2 - 1) * in_width + (ox * 2 - 1);
                    const row1 = row0 + in_width;
                    const row2 = row1 + in_width;

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
                        const weight_base = weights_channel_base + ic * 9;
                        acc += input_channel[row0] * weights.data[weight_base];
                        acc += input_channel[row0 + 1] * weights.data[weight_base + 1];
                        acc += input_channel[row0 + 2] * weights.data[weight_base + 2];
                        acc += input_channel[row1] * weights.data[weight_base + 3];
                        acc += input_channel[row1 + 1] * weights.data[weight_base + 4];
                        acc += input_channel[row1 + 2] * weights.data[weight_base + 5];
                        acc += input_channel[row2] * weights.data[weight_base + 6];
                        acc += input_channel[row2 + 1] * weights.data[weight_base + 7];
                        acc += input_channel[row2 + 2] * weights.data[weight_base + 8];
                    }

                    output.data[output_index] = acc;
                }

                for (interior_w_end..expected_w) |ox| {
                    output.data[output_row_base + ox] = conv2d3x3Pad1Stride2Point(
                        input,
                        weights,
                        bias_value,
                        input_batch_base,
                        weights_channel_base,
                        oy,
                        ox,
                    );
                }
            }
        }
    }
}

fn conv2d3x3Pad1Stride2Point(
    input: *const Tensor,
    weights: *const Tensor,
    bias_value: f32,
    input_batch_base: usize,
    weights_channel_base: usize,
    oy: usize,
    ox: usize,
) f32 {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const input_plane = in_height * in_width;
    const base_y = @as(isize, @intCast(oy * 2)) - 1;
    const base_x = @as(isize, @intCast(ox * 2)) - 1;

    var acc: f32 = bias_value;
    for (0..in_channels) |ic| {
        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
        const weight_base = weights_channel_base + ic * 9;
        const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
        const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
        const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
        const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

        var iy = y_start;
        while (iy < y_end) : (iy += 1) {
            const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
            const input_row_base = iy * in_width;
            const weight_row_base = weight_base + ky * 3;
            var ix = x_start;
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                acc += input_channel[input_row_base + ix] * weights.data[weight_row_base + kx];
            }
        }
    }
    return acc;
}


fn conv2dGeneralParallel(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    thread_count: usize,
) OpError!void {
    var threads: [max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2dGeneralRange(input, weights, bias, output, options, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2dGeneralWorker, .{
                Conv2DTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .options = options,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                try conv2dGeneralRange(input, weights, bias, output, options, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

fn conv2dGeneralRange(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
) OpError!void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_channels = weights.shape[0];
    const kernel_in_channels = weights.shape[1];
    const kernel_h = weights.shape[2];
    const kernel_w = weights.shape[3];
    const expected_h = ((in_height + 2 * options.pad_h - kernel_h) / options.stride_h) + 1;
    const expected_w = ((in_width + 2 * options.pad_w - kernel_w) / options.stride_w) + 1;

    const in_per_group = in_channels / options.groups;
    const out_per_group = out_channels / options.groups;
    const input_channel_plane = in_height * in_width;
    const output_channel_plane = expected_h * expected_w;
    const weights_in_plane = kernel_h * kernel_w;

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * input_channel_plane;
        const output_batch_base = n * out_channels * output_channel_plane;
        for (oc_start..oc_end) |oc| {
            const group_idx = oc / out_per_group;
            const in_channel_start = group_idx * in_per_group;
            const output_channel_base = output_batch_base + oc * output_channel_plane;
            const weights_channel_base = oc * kernel_in_channels * weights_in_plane;

            for (0..expected_h) |oy| {
                const output_row_base = output_channel_base + oy * expected_w;
                for (0..expected_w) |ox| {
                    var acc: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                    const base_y = @as(isize, @intCast(oy * options.stride_h)) - @as(isize, @intCast(options.pad_h));
                    const base_x = @as(isize, @intCast(ox * options.stride_w)) - @as(isize, @intCast(options.pad_w));

                    for (0..in_per_group) |ic_local| {
                        const ic = in_channel_start + ic_local;
                        const input_channel_base = input_batch_base + ic * input_channel_plane;
                        const weights_in_base = weights_channel_base + ic_local * weights_in_plane;
                        for (0..kernel_h) |ky| {
                            const in_y = base_y + @as(isize, @intCast(ky));
                            if (in_y < 0 or in_y >= @as(isize, @intCast(in_height))) continue;
                            const input_row_base = input_channel_base + @as(usize, @intCast(in_y)) * in_width;
                            const weights_row_base = weights_in_base + ky * kernel_w;

                            for (0..kernel_w) |kx| {
                                const in_x = base_x + @as(isize, @intCast(kx));
                                if (in_x < 0 or in_x >= @as(isize, @intCast(in_width))) continue;
                                const input_value = input.data[input_row_base + @as(usize, @intCast(in_x))];
                                const weight_value = weights.data[weights_row_base + kx];
                                acc += input_value * weight_value;
                            }
                        }
                    }
                    output.data[output_row_base + ox] = acc;
                }
            }
        }
    }
}

fn conv2dPointwise(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    groups: usize,
) OpError!void {
    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const plane = input.shape[2] * input.shape[3];
    const workload = batch * out_channels * plane * (input.shape[1] / groups);
    const thread_count = chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2dPointwiseParallel(input, weights, bias, output, groups, thread_count);
    }

    return conv2dPointwiseRange(input, weights, bias, output, groups, 0, out_channels);
}

fn conv2dPointwiseParallel(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    groups: usize,
    thread_count: usize,
) OpError!void {
    var threads: [max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2dPointwiseRange(input, weights, bias, output, groups, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2dPointwiseWorker, .{
                Conv2DPointwiseTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .groups = groups,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                try conv2dPointwiseRange(input, weights, bias, output, groups, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

fn conv2dPointwiseRange(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    groups: usize,
    oc_start: usize,
    oc_end: usize,
) OpError!void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];
    const out_channels = weights.shape[0];
    const in_per_group = in_channels / groups;
    const out_per_group = out_channels / groups;
    const plane = height * width;

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * plane;
        const output_batch_base = n * out_channels * plane;
        var oc = oc_start;
        while (oc < oc_end) {
            const group_idx = oc / out_per_group;
            const in_channel_start = group_idx * in_per_group;
            const pairable = oc + 1 < oc_end and (oc + 1) / out_per_group == group_idx;

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                @memset(out0_slice, bias0);
                @memset(out1_slice, bias1);

                const weight0_base = oc * in_per_group;
                const weight1_base = (oc + 1) * in_per_group;
                for (0..in_per_group) |ic_local| {
                    const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                    const weight0 = weights.data[weight0_base + ic_local];
                    const weight1 = weights.data[weight1_base + ic_local];
                    for (out0_slice, out1_slice, input_slice) |*dst0, *dst1, src| {
                        dst0.* += src * weight0;
                        dst1.* += src * weight1;
                    }
                }
                oc += 2;
                continue;
            }

            const out_slice = output.data[output_batch_base + oc * plane ..][0..plane];
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            @memset(out_slice, bias_value);

            const weight_base = oc * in_per_group;
            for (0..in_per_group) |ic_local| {
                const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                const weight_value = weights.data[weight_base + ic_local];
                for (out_slice, input_slice) |*dst, src| {
                    dst.* += src * weight_value;
                }
            }
            oc += 1;
        }
    }
}

const Conv2DTask = struct {
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
};

fn conv2dGeneralWorker(task: Conv2DTask) void {
    conv2dGeneralRange(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
}

fn conv2d3x3Pad1Worker(task: Conv2DTask) void {
    conv2d3x3Pad1Range(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
}

fn conv2d3x3Pad1Stride2Worker(task: Conv2DTask) void {
    conv2d3x3Pad1Stride2Range(task.input, task.weights, task.bias, task.output, task.oc_start, task.oc_end) catch unreachable;
}

const Conv2DPointwiseTask = struct {
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    groups: usize,
    oc_start: usize,
    oc_end: usize,
};

fn conv2dPointwiseWorker(task: Conv2DPointwiseTask) void {
    conv2dPointwiseRange(task.input, task.weights, task.bias, task.output, task.groups, task.oc_start, task.oc_end) catch unreachable;
}

fn chooseConvThreadCount(workload: usize, out_channels: usize) usize {
    if (out_channels < 2 or workload < conv_parallel_min_workload) return 1;
    return @min(out_channels, 4);
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
    try conv2dGeneralRange(&input, &weights, &bias_values, &general, .{
        .stride_h = 2,
        .stride_w = 2,
        .pad_h = 1,
        .pad_w = 1,
    }, 0, 3);

    for (fast.data, general.data) |actual, expected| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}
