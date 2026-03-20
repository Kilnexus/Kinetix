const std = @import("std");
const common = @import("common.zig");
const tasks = @import("tasks.zig");

pub fn conv2d3x3Pad1(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
) common.OpError!void {
    if (options.stride_h == 2 and options.stride_w == 2) {
        return conv2d3x3Pad1Stride2(input, weights, bias, output, options);
    }

    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2d3x3Pad1Parallel(input, weights, bias, output, options, thread_count);
    }

    return conv2d3x3Pad1Range(input, weights, bias, output, options, 0, out_channels);
}

fn conv2d3x3Pad1Stride2(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
) common.OpError!void {
    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2d3x3Pad1Stride2Parallel(input, weights, bias, output, options, thread_count);
    }

    return conv2d3x3Pad1Stride2Range(input, weights, bias, output, options, 0, out_channels);
}

fn conv2d3x3Pad1Parallel(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    thread_count: usize,
) common.OpError!void {
    var threads: [common.max_supported_conv_threads - 1]std.Thread = undefined;
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
                tasks.Conv2DTask{
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
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    thread_count: usize,
) common.OpError!void {
    var threads: [common.max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2d3x3Pad1Stride2Range(input, weights, bias, output, options, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2d3x3Pad1Stride2Worker, .{
                tasks.Conv2DTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .options = options,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                try conv2d3x3Pad1Stride2Range(input, weights, bias, output, options, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

fn conv2d3x3Pad1Range(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
) common.OpError!void {
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
                            const src8: common.F32xN = .{ v00, v01, v02, v10, v11, v12, v20, v21 };

                            acc0 += common.dotF32xN(src8, common.loadF32xN(weights.data, weight0_base));
                            acc0 += v22 * weights.data[weight0_base + 8];

                            acc1 += common.dotF32xN(src8, common.loadF32xN(weights.data, weight1_base));
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

                    output.data[output0_row_base + ox] = common.maybeApplySilu(acc0, options.apply_silu);
                    output.data[output1_row_base + ox] = common.maybeApplySilu(acc1, options.apply_silu);
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
                            const v00 = input_channel[row0];
                            const v01 = input_channel[row0 + 1];
                            const v02 = input_channel[row0 + 2];
                            const v10 = input_channel[row1];
                            const v11 = input_channel[row1 + 1];
                            const v12 = input_channel[row1 + 2];
                            const v20 = input_channel[row2];
                            const v21 = input_channel[row2 + 1];
                            const v22 = input_channel[row2 + 2];
                            const src8: common.F32xN = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                            acc += common.dotF32xN(src8, common.loadF32xN(weights.data, weight_base));
                            acc += v22 * weights.data[weight_base + 8];
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

                    output.data[output_row_base + ox] = common.maybeApplySilu(acc, options.apply_silu);
                }
            }
        }
    }
}

fn conv2d3x3Pad1Stride2Range(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
) common.OpError!void {
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
                        output.data[output0_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias0, input_batch_base, weights0_channel_base, oy, ox), options.apply_silu);
                        output.data[output1_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias1, input_batch_base, weights1_channel_base, oy, ox), options.apply_silu);
                    }
                    continue;
                }

                output.data[output0_row_base] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias0, input_batch_base, weights0_channel_base, oy, 0), options.apply_silu);
                output.data[output1_row_base] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias1, input_batch_base, weights1_channel_base, oy, 0), options.apply_silu);

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
                        const src8: common.F32xN = .{ v00, v01, v02, v10, v11, v12, v20, v21 };

                        acc0 += common.dotF32xN(src8, common.loadF32xN(weights.data, weight0_base));
                        acc0 += v22 * weights.data[weight0_base + 8];

                        acc1 += common.dotF32xN(src8, common.loadF32xN(weights.data, weight1_base));
                        acc1 += v22 * weights.data[weight1_base + 8];
                    }

                    output.data[output0_index] = common.maybeApplySilu(acc0, options.apply_silu);
                    output.data[output1_index] = common.maybeApplySilu(acc1, options.apply_silu);
                }

                for (interior_w_end..expected_w) |ox| {
                    output.data[output0_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias0, input_batch_base, weights0_channel_base, oy, ox), options.apply_silu);
                    output.data[output1_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias1, input_batch_base, weights1_channel_base, oy, ox), options.apply_silu);
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
                        output.data[output_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, ox), options.apply_silu);
                    }
                    continue;
                }

                output.data[output_row_base] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, 0), options.apply_silu);

                for (1..interior_w_end) |ox| {
                    const output_index = output_row_base + ox;
                    var acc: f32 = bias_value;
                    const row0 = (oy * 2 - 1) * in_width + (ox * 2 - 1);
                    const row1 = row0 + in_width;
                    const row2 = row1 + in_width;

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
                        const weight_base = weights_channel_base + ic * 9;
                        const v00 = input_channel[row0];
                        const v01 = input_channel[row0 + 1];
                        const v02 = input_channel[row0 + 2];
                        const v10 = input_channel[row1];
                        const v11 = input_channel[row1 + 1];
                        const v12 = input_channel[row1 + 2];
                        const v20 = input_channel[row2];
                        const v21 = input_channel[row2 + 1];
                        const v22 = input_channel[row2 + 2];
                        const src8: common.F32xN = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                        acc += common.dotF32xN(src8, common.loadF32xN(weights.data, weight_base));
                        acc += v22 * weights.data[weight_base + 8];
                    }

                    output.data[output_index] = common.maybeApplySilu(acc, options.apply_silu);
                }

                for (interior_w_end..expected_w) |ox| {
                    output.data[output_row_base + ox] = common.maybeApplySilu(conv2d3x3Pad1Stride2Point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, ox), options.apply_silu);
                }
            }
        }
    }
}

fn conv2d3x3Pad1Stride2Point(
    input: *const common.Tensor,
    weights: *const common.Tensor,
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
            if (x_end - x_start == 3) {
                const v0 = input_channel[input_row_base + x_start];
                const v1 = input_channel[input_row_base + x_start + 1];
                const v2 = input_channel[input_row_base + x_start + 2];
                const w0 = weights.data[weight_row_base + @as(usize, @intCast(@as(isize, @intCast(x_start)) - base_x))];
                const w1 = weights.data[weight_row_base + @as(usize, @intCast(@as(isize, @intCast(x_start + 1)) - base_x))];
                const w2 = weights.data[weight_row_base + @as(usize, @intCast(@as(isize, @intCast(x_start + 2)) - base_x))];
                acc += v0 * w0 + v1 * w1 + v2 * w2;
                continue;
            }
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                acc += input_channel[input_row_base + ix] * weights.data[weight_row_base + kx];
            }
        }
    }
    return acc;
}

fn conv2d3x3Pad1Worker(task: tasks.Conv2DTask) void {
    conv2d3x3Pad1Range(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
}

fn conv2d3x3Pad1Stride2Worker(task: tasks.Conv2DTask) void {
    conv2d3x3Pad1Stride2Range(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
}
