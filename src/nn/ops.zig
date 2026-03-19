const std = @import("std");
const tensor_mod = @import("tensor");

pub const Tensor = tensor_mod.Tensor;

pub const OpError = error{
    ShapeMismatch,
    InvalidOutputShape,
    InvalidGroups,
    InvalidTensorRank,
};

pub const Conv2DOptions = struct {
    stride_h: usize = 1,
    stride_w: usize = 1,
    pad_h: usize = 0,
    pad_w: usize = 0,
    groups: usize = 1,
};

const max_supported_conv_threads = 4;
const conv_thread_cap = 4;
const conv_parallel_min_workload = 2_000_000;

pub fn siluInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = x / (1.0 + @exp(-x));
    }
}

pub fn sigmoidInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = 1.0 / (1.0 + @exp(-x));
    }
}

pub fn add(output: *Tensor, lhs: *const Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(lhs) or !lhs.sameShape(rhs)) return OpError.ShapeMismatch;
    for (output.data, lhs.data, rhs.data) |*out, left, right| out.* = left + right;
}

pub fn upsampleNearest(
    input: *const Tensor,
    output: *Tensor,
    scale_h: usize,
    scale_w: usize,
) OpError!void {
    if (output.shape[0] != input.shape[0] or output.shape[1] != input.shape[1]) {
        return OpError.ShapeMismatch;
    }
    if (output.shape[2] != input.shape[2] * scale_h or output.shape[3] != input.shape[3] * scale_w) {
        return OpError.InvalidOutputShape;
    }

    const in_plane = input.shape[2] * input.shape[3];
    const out_plane = output.shape[2] * output.shape[3];

    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input.shape[1] * in_plane;
        const output_batch_base = n * output.shape[1] * out_plane;
        for (0..input.shape[1]) |c| {
            const input_channel = input.data[input_batch_base + c * in_plane ..][0..in_plane];
            const output_channel = output.data[output_batch_base + c * out_plane ..][0..out_plane];
            for (0..input.shape[2]) |iy| {
                const input_row = input_channel[iy * input.shape[3] ..][0..input.shape[3]];
                for (0..scale_h) |dy| {
                    const output_row = output_channel[(iy * scale_h + dy) * output.shape[3] ..][0..output.shape[3]];
                    for (input_row, 0..) |value, ix| {
                        const out_x = ix * scale_w;
                        for (0..scale_w) |dx| {
                            output_row[out_x + dx] = value;
                        }
                    }
                }
            }
        }
    }
}

pub fn concatChannels(inputs: []const *const Tensor, output: *Tensor) OpError!void {
    if (inputs.len == 0) return OpError.ShapeMismatch;

    const batch = inputs[0].shape[0];
    const height = inputs[0].shape[2];
    const width = inputs[0].shape[3];
    var total_channels: usize = 0;

    for (inputs) |input| {
        if (input.shape[0] != batch or input.shape[2] != height or input.shape[3] != width) {
            return OpError.ShapeMismatch;
        }
        total_channels += input.shape[1];
    }

    if (output.shape[0] != batch or output.shape[1] != total_channels or output.shape[2] != height or output.shape[3] != width) {
        return OpError.InvalidOutputShape;
    }

    var channel_offset: usize = 0;
    for (inputs) |input| {
        const plane = height * width;
        for (0..batch) |n| {
            const input_batch_base = n * input.shape[1] * plane;
            const output_batch_base = n * output.shape[1] * plane;
            for (0..input.shape[1]) |c| {
                const src = input.data[input_batch_base + c * plane ..][0..plane];
                const dst = output.data[output_batch_base + (channel_offset + c) * plane ..][0..plane];
                @memcpy(dst, src);
            }
        }
        channel_offset += input.shape[1];
    }
}

pub fn copyChannelRange(
    input: *const Tensor,
    input_channel_start: usize,
    channel_count: usize,
    output: *Tensor,
    output_channel_start: usize,
) OpError!void {
    if (input.shape[0] != output.shape[0] or input.shape[2] != output.shape[2] or input.shape[3] != output.shape[3]) {
        return OpError.ShapeMismatch;
    }
    if (input_channel_start + channel_count > input.shape[1] or output_channel_start + channel_count > output.shape[1]) {
        return OpError.InvalidOutputShape;
    }

    const plane = input.shape[2] * input.shape[3];
    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input.shape[1] * plane;
        const output_batch_base = n * output.shape[1] * plane;
        for (0..channel_count) |c| {
            const src = input.data[input_batch_base + (input_channel_start + c) * plane ..][0..plane];
            const dst = output.data[output_batch_base + (output_channel_start + c) * plane ..][0..plane];
            @memcpy(dst, src);
        }
    }
}

pub fn maxPool2d(
    input: *const Tensor,
    output: *Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) OpError!void {
    if (input.shape[0] != output.shape[0] or input.shape[1] != output.shape[1]) {
        return OpError.ShapeMismatch;
    }

    const expected_h = ((input.shape[2] + 2 * pad_h - kernel_h) / stride_h) + 1;
    const expected_w = ((input.shape[3] + 2 * pad_w - kernel_w) / stride_w) + 1;
    if (output.shape[2] != expected_h or output.shape[3] != expected_w) {
        return OpError.InvalidOutputShape;
    }

    for (0..input.shape[0]) |n| {
        for (0..input.shape[1]) |c| {
            for (0..output.shape[2]) |oy| {
                for (0..output.shape[3]) |ox| {
                    var max_value = -std.math.inf(f32);
                    const base_y = @as(isize, @intCast(oy * stride_h)) - @as(isize, @intCast(pad_h));
                    const base_x = @as(isize, @intCast(ox * stride_w)) - @as(isize, @intCast(pad_w));

                    for (0..kernel_h) |ky| {
                        const in_y = base_y + @as(isize, @intCast(ky));
                        if (in_y < 0 or in_y >= @as(isize, @intCast(input.shape[2]))) continue;

                        for (0..kernel_w) |kx| {
                            const in_x = base_x + @as(isize, @intCast(kx));
                            if (in_x < 0 or in_x >= @as(isize, @intCast(input.shape[3]))) continue;

                            const value = input.get(n, c, @intCast(in_y), @intCast(in_x));
                            if (value > max_value) max_value = value;
                        }
                    }
                    output.set(n, c, oy, ox, max_value);
                }
            }
        }
    }
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
        for (oc_start..oc_end) |oc| {
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
        for (oc_start..oc_end) |oc| {
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
        for (oc_start..oc_end) |oc| {
            const group_idx = oc / out_per_group;
            const in_channel_start = group_idx * in_per_group;
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
    return @min(out_channels, conv_thread_cap);
}

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

test "concat channels preserves order" {
    const testing = std.testing;

    var lhs = try Tensor.init(testing.allocator, 1, 1, 1, 2);
    defer lhs.deinit();
    lhs.data[0] = 1.0;
    lhs.data[1] = 2.0;

    var rhs = try Tensor.init(testing.allocator, 1, 2, 1, 2);
    defer rhs.deinit();
    rhs.data[0] = 3.0;
    rhs.data[1] = 4.0;
    rhs.data[2] = 5.0;
    rhs.data[3] = 6.0;

    var output = try Tensor.init(testing.allocator, 1, 3, 1, 2);
    defer output.deinit();

    const inputs = [_]*const Tensor{ &lhs, &rhs };
    try concatChannels(&inputs, &output);

    try testing.expectApproxEqAbs(@as(f32, 1.0), output.get(0, 0, 0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), output.get(0, 1, 0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output.get(0, 2, 0, 1), 1e-6);
}

test "upsample nearest duplicates pixels" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 1, 1, 2);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;

    var output = try Tensor.init(testing.allocator, 1, 1, 2, 4);
    defer output.deinit();

    try upsampleNearest(&input, &output, 2, 2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output.get(0, 0, 0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output.get(0, 0, 1, 3), 1e-6);
}

test "maxpool selects local maximum" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 1, 2, 2);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 9.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    var output = try Tensor.init(testing.allocator, 1, 1, 1, 1);
    defer output.deinit();

    try maxPool2d(&input, &output, 2, 2, 2, 2, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 9.0), output.data[0], 1e-6);
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
