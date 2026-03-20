const std = @import("std");
const common = @import("common.zig");
const tasks = @import("tasks.zig");

pub fn conv2dPointwise(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    groups: usize,
    apply_silu: bool,
) common.OpError!void {
    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const plane = input.shape[2] * input.shape[3];
    const workload = batch * out_channels * plane * (input.shape[1] / groups);
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2dPointwiseParallel(input, weights, bias, output, groups, thread_count, apply_silu);
    }

    return conv2dPointwiseRange(input, weights, bias, output, groups, 0, out_channels, apply_silu);
}

fn conv2dPointwiseParallel(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    groups: usize,
    thread_count: usize,
    apply_silu: bool,
) common.OpError!void {
    var threads: [common.max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try conv2dPointwiseRange(input, weights, bias, output, groups, oc_start, oc_end, apply_silu);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2dPointwiseWorker, .{
                tasks.Conv2DPointwiseTask{
                    .input = input,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .groups = groups,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                    .apply_silu = apply_silu,
                },
            }) catch {
                try conv2dPointwiseRange(input, weights, bias, output, groups, oc_start, oc_end, apply_silu);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

pub fn conv2dPointwiseRange(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    groups: usize,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
) common.OpError!void {
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
                    const w0 = @as(common.F32xN, @splat(weight0));
                    const w1 = @as(common.F32xN, @splat(weight1));
                    var i: usize = 0;
                    while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                        const src = common.loadF32xN(input_slice, i);
                        const acc0 = common.loadF32xN(out0_slice, i) + src * w0;
                        const acc1 = common.loadF32xN(out1_slice, i) + src * w1;
                        common.storeF32xN(out0_slice, i, acc0);
                        common.storeF32xN(out1_slice, i, acc1);
                    }
                    while (i < plane) : (i += 1) {
                        const src = input_slice[i];
                        out0_slice[i] += src * weight0;
                        out1_slice[i] += src * weight1;
                    }
                }
                if (apply_silu) {
                    for (out0_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out1_slice) |*dst| dst.* = common.siluValue(dst.*);
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
                const w = @as(common.F32xN, @splat(weight_value));
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    const src = common.loadF32xN(input_slice, i);
                    const acc = common.loadF32xN(out_slice, i) + src * w;
                    common.storeF32xN(out_slice, i, acc);
                }
                while (i < plane) : (i += 1) {
                    out_slice[i] += input_slice[i] * weight_value;
                }
            }
            if (apply_silu) {
                for (out_slice) |*dst| dst.* = common.siluValue(dst.*);
            }
            oc += 1;
        }
    }
}

fn conv2dPointwiseWorker(task: tasks.Conv2DPointwiseTask) void {
    conv2dPointwiseRange(task.input, task.weights, task.bias, task.output, task.groups, task.oc_start, task.oc_end, task.apply_silu) catch unreachable;
}
