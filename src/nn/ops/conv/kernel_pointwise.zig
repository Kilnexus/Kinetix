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

pub fn conv2dPointwiseConcat(
    inputs: []const *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    apply_silu: bool,
) common.OpError!void {
    if (inputs.len == 0) return common.OpError.ShapeMismatch;

    const batch = inputs[0].shape[0];
    const height = inputs[0].shape[2];
    const width = inputs[0].shape[3];
    const out_channels = weights.shape[0];
    const plane = height * width;

    var total_in_channels: usize = 0;
    var offsets_stack: [16]usize = undefined;
    if (inputs.len > offsets_stack.len) return common.OpError.InvalidTensorRank;
    const offsets = offsets_stack[0..inputs.len];

    for (inputs, 0..) |input, index| {
        if (input.shape[0] != batch or input.shape[2] != height or input.shape[3] != width) {
            return common.OpError.ShapeMismatch;
        }
        offsets[index] = total_in_channels;
        total_in_channels += input.shape[1];
    }

    if (weights.shape[1] != total_in_channels or weights.shape[2] != 1 or weights.shape[3] != 1) {
        return common.OpError.ShapeMismatch;
    }
    if (output.shape[0] != batch or output.shape[1] != out_channels or output.shape[2] != height or output.shape[3] != width) {
        return common.OpError.InvalidOutputShape;
    }
    if (bias) |bias_values| {
        if (bias_values.len != out_channels) return common.OpError.ShapeMismatch;
    }

    const workload = batch * out_channels * plane * total_in_channels;
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return conv2dPointwiseConcatParallel(inputs, offsets, weights, bias, output, thread_count, apply_silu);
    }

    return conv2dPointwiseConcatRange(inputs, offsets, weights, bias, output, 0, out_channels, apply_silu);
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

fn conv2dPointwiseConcatParallel(
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
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
            try conv2dPointwiseConcatRange(inputs, input_channel_offsets, weights, bias, output, oc_start, oc_end, apply_silu);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, conv2dPointwiseConcatWorker, .{
                tasks.Conv2DPointwiseConcatTask{
                    .inputs = inputs,
                    .input_channel_offsets = input_channel_offsets,
                    .weights = weights,
                    .bias = bias,
                    .output = output,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                    .apply_silu = apply_silu,
                },
            }) catch {
                try conv2dPointwiseConcatRange(inputs, input_channel_offsets, weights, bias, output, oc_start, oc_end, apply_silu);
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
            const quadable = oc + 3 < oc_end and
                (oc + 3) / out_per_group == group_idx;
            const pairable = oc + 1 < oc_end and (oc + 1) / out_per_group == group_idx;

            if (quadable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const out2_slice = output.data[output_batch_base + (oc + 2) * plane ..][0..plane];
                const out3_slice = output.data[output_batch_base + (oc + 3) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                const bias2: f32 = if (bias) |bias_values| bias_values[oc + 2] else 0.0;
                const bias3: f32 = if (bias) |bias_values| bias_values[oc + 3] else 0.0;
                @memset(out0_slice, bias0);
                @memset(out1_slice, bias1);
                @memset(out2_slice, bias2);
                @memset(out3_slice, bias3);

                const weight0_base = oc * in_per_group;
                const weight1_base = (oc + 1) * in_per_group;
                const weight2_base = (oc + 2) * in_per_group;
                const weight3_base = (oc + 3) * in_per_group;
                for (0..in_per_group) |ic_local| {
                    const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                    const weight0 = weights.data[weight0_base + ic_local];
                    const weight1 = weights.data[weight1_base + ic_local];
                    const weight2 = weights.data[weight2_base + ic_local];
                    const weight3 = weights.data[weight3_base + ic_local];
                    const w0 = @as(common.F32xN, @splat(weight0));
                    const w1 = @as(common.F32xN, @splat(weight1));
                    const w2 = @as(common.F32xN, @splat(weight2));
                    const w3 = @as(common.F32xN, @splat(weight3));
                    var i: usize = 0;
                    while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                        const src = common.loadF32xN(input_slice, i);
                        common.storeF32xN(out0_slice, i, common.loadF32xN(out0_slice, i) + src * w0);
                        common.storeF32xN(out1_slice, i, common.loadF32xN(out1_slice, i) + src * w1);
                        common.storeF32xN(out2_slice, i, common.loadF32xN(out2_slice, i) + src * w2);
                        common.storeF32xN(out3_slice, i, common.loadF32xN(out3_slice, i) + src * w3);
                    }
                    while (i < plane) : (i += 1) {
                        const src = input_slice[i];
                        out0_slice[i] += src * weight0;
                        out1_slice[i] += src * weight1;
                        out2_slice[i] += src * weight2;
                        out3_slice[i] += src * weight3;
                    }
                }
                if (apply_silu) {
                    for (out0_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out1_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out2_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out3_slice) |*dst| dst.* = common.siluValue(dst.*);
                }
                oc += 4;
                continue;
            }

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

pub fn conv2dPointwiseConcatRange(
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
) common.OpError!void {
    const batch = output.shape[0];
    const out_channels = weights.shape[0];
    const plane = output.shape[2] * output.shape[3];

    for (0..batch) |n| {
        const output_batch_base = n * out_channels * plane;
        var oc = oc_start;
        while (oc < oc_end) {
            const quadable = oc + 3 < oc_end;
            const pairable = oc + 1 < oc_end;

            if (quadable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const out2_slice = output.data[output_batch_base + (oc + 2) * plane ..][0..plane];
                const out3_slice = output.data[output_batch_base + (oc + 3) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                const bias2: f32 = if (bias) |bias_values| bias_values[oc + 2] else 0.0;
                const bias3: f32 = if (bias) |bias_values| bias_values[oc + 3] else 0.0;
                @memset(out0_slice, bias0);
                @memset(out1_slice, bias1);
                @memset(out2_slice, bias2);
                @memset(out3_slice, bias3);

                const weight0_base = oc * weights.shape[1];
                const weight1_base = (oc + 1) * weights.shape[1];
                const weight2_base = (oc + 2) * weights.shape[1];
                const weight3_base = (oc + 3) * weights.shape[1];
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    for (0..input.shape[1]) |ic| {
                        const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                        const weight_index = channel_offset + ic;
                        const weight0 = weights.data[weight0_base + weight_index];
                        const weight1 = weights.data[weight1_base + weight_index];
                        const weight2 = weights.data[weight2_base + weight_index];
                        const weight3 = weights.data[weight3_base + weight_index];
                        const w0 = @as(common.F32xN, @splat(weight0));
                        const w1 = @as(common.F32xN, @splat(weight1));
                        const w2 = @as(common.F32xN, @splat(weight2));
                        const w3 = @as(common.F32xN, @splat(weight3));
                        var i: usize = 0;
                        while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                            const src = common.loadF32xN(input_slice, i);
                            common.storeF32xN(out0_slice, i, common.loadF32xN(out0_slice, i) + src * w0);
                            common.storeF32xN(out1_slice, i, common.loadF32xN(out1_slice, i) + src * w1);
                            common.storeF32xN(out2_slice, i, common.loadF32xN(out2_slice, i) + src * w2);
                            common.storeF32xN(out3_slice, i, common.loadF32xN(out3_slice, i) + src * w3);
                        }
                        while (i < plane) : (i += 1) {
                            const src = input_slice[i];
                            out0_slice[i] += src * weight0;
                            out1_slice[i] += src * weight1;
                            out2_slice[i] += src * weight2;
                            out3_slice[i] += src * weight3;
                        }
                    }
                }
                if (apply_silu) {
                    for (out0_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out1_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out2_slice) |*dst| dst.* = common.siluValue(dst.*);
                    for (out3_slice) |*dst| dst.* = common.siluValue(dst.*);
                }
                oc += 4;
                continue;
            }

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                @memset(out0_slice, bias0);
                @memset(out1_slice, bias1);

                const weight0_base = oc * weights.shape[1];
                const weight1_base = (oc + 1) * weights.shape[1];
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    for (0..input.shape[1]) |ic| {
                        const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                        const weight_index = channel_offset + ic;
                        const weight0 = weights.data[weight0_base + weight_index];
                        const weight1 = weights.data[weight1_base + weight_index];
                        const w0 = @as(common.F32xN, @splat(weight0));
                        const w1 = @as(common.F32xN, @splat(weight1));
                        var i: usize = 0;
                        while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                            const src = common.loadF32xN(input_slice, i);
                            common.storeF32xN(out0_slice, i, common.loadF32xN(out0_slice, i) + src * w0);
                            common.storeF32xN(out1_slice, i, common.loadF32xN(out1_slice, i) + src * w1);
                        }
                        while (i < plane) : (i += 1) {
                            const src = input_slice[i];
                            out0_slice[i] += src * weight0;
                            out1_slice[i] += src * weight1;
                        }
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

            const weight_base = oc * weights.shape[1];
            for (inputs, input_channel_offsets) |input, channel_offset| {
                const input_batch_base = n * input.shape[1] * plane;
                for (0..input.shape[1]) |ic| {
                    const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                    const weight_value = weights.data[weight_base + channel_offset + ic];
                    const w = @as(common.F32xN, @splat(weight_value));
                    var i: usize = 0;
                    while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                        const src = common.loadF32xN(input_slice, i);
                        common.storeF32xN(out_slice, i, common.loadF32xN(out_slice, i) + src * w);
                    }
                    while (i < plane) : (i += 1) {
                        out_slice[i] += input_slice[i] * weight_value;
                    }
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

fn conv2dPointwiseConcatWorker(task: tasks.Conv2DPointwiseConcatTask) void {
    conv2dPointwiseConcatRange(
        task.inputs,
        task.input_channel_offsets,
        task.weights,
        task.bias,
        task.output,
        task.oc_start,
        task.oc_end,
        task.apply_silu,
    ) catch unreachable;
}
