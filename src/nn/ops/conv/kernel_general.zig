const std = @import("std");
const common = @import("common.zig");
const tasks = @import("tasks.zig");

pub fn conv2dGeneralParallel(
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
            try conv2dGeneralRange(input, weights, bias, output, options, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2dGeneralWorker, .{
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
                try conv2dGeneralRange(input, weights, bias, output, options, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

pub fn conv2dGeneralRange(
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
                    output.data[output_row_base + ox] = common.maybeApplySilu(acc, options.apply_silu);
                }
            }
        }
    }
}

fn conv2dGeneralWorker(task: tasks.Conv2DTask) void {
    conv2dGeneralRange(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
}
