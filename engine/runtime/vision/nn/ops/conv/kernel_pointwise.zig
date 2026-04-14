const std = @import("std");
const common = @import("common.zig");
const parallel = @import("parallel.zig");
const tasks = @import("tasks.zig");

const PointwisePackKey = struct {
    ptr: usize,
    out_channels: usize,
    in_per_group: usize,
    groups: usize,
};

const PackedPointwiseWeights = struct {
    out_channels: usize,
    in_per_group: usize,
    groups: usize,
    out_per_group: usize,
    data: []f32,
};

var pointwise_pack_cache_mutex: std.Thread.Mutex = .{};
var pointwise_pack_cache: std.AutoHashMapUnmanaged(PointwisePackKey, PackedPointwiseWeights) = .{};
const pointwise_pack_tl_capacity = 16;
threadlocal var pointwise_pack_tl_len: usize = 0;
threadlocal var pointwise_pack_tl_next: usize = 0;
threadlocal var pointwise_pack_tl_keys: [pointwise_pack_tl_capacity]PointwisePackKey = undefined;
threadlocal var pointwise_pack_tl_values: [pointwise_pack_tl_capacity]PackedPointwiseWeights = undefined;

fn pointwisePackKeyEqual(lhs: PointwisePackKey, rhs: PointwisePackKey) bool {
    return lhs.ptr == rhs.ptr and
        lhs.out_channels == rhs.out_channels and
        lhs.in_per_group == rhs.in_per_group and
        lhs.groups == rhs.groups;
}

fn getThreadLocalPackedPointwiseWeights(key: PointwisePackKey) ?PackedPointwiseWeights {
    for (0..pointwise_pack_tl_len) |index| {
        if (pointwisePackKeyEqual(pointwise_pack_tl_keys[index], key)) {
            return pointwise_pack_tl_values[index];
        }
    }
    return null;
}

fn putThreadLocalPackedPointwiseWeights(
    key: PointwisePackKey,
    pack_weights: PackedPointwiseWeights,
) void {
    for (0..pointwise_pack_tl_len) |index| {
        if (pointwisePackKeyEqual(pointwise_pack_tl_keys[index], key)) {
            pointwise_pack_tl_values[index] = pack_weights;
            return;
        }
    }

    if (pointwise_pack_tl_len < pointwise_pack_tl_capacity) {
        const index = pointwise_pack_tl_len;
        pointwise_pack_tl_keys[index] = key;
        pointwise_pack_tl_values[index] = pack_weights;
        pointwise_pack_tl_len += 1;
        return;
    }

    const replace_index = pointwise_pack_tl_next;
    pointwise_pack_tl_keys[replace_index] = key;
    pointwise_pack_tl_values[replace_index] = pack_weights;
    pointwise_pack_tl_next = (pointwise_pack_tl_next + 1) % pointwise_pack_tl_capacity;
}

fn getPackedPointwiseWeights(
    weights: *const common.Tensor,
    groups: usize,
) ?PackedPointwiseWeights {
    if (groups == 0) return null;
    const out_channels = weights.shape[0];
    const in_per_group = weights.shape[1];
    if (out_channels == 0 or in_per_group == 0) return null;
    if (out_channels % groups != 0) return null;

    const key = PointwisePackKey{
        .ptr = @intFromPtr(weights.data.ptr),
        .out_channels = out_channels,
        .in_per_group = in_per_group,
        .groups = groups,
    };

    if (getThreadLocalPackedPointwiseWeights(key)) |cached| return cached;

    pointwise_pack_cache_mutex.lock();
    defer pointwise_pack_cache_mutex.unlock();

    if (pointwise_pack_cache.get(key)) |cached| {
        putThreadLocalPackedPointwiseWeights(key, cached);
        return cached;
    }

    const pack_weights = buildPackedPointwiseWeights(weights, groups) orelse return null;
    pointwise_pack_cache.put(std.heap.page_allocator, key, pack_weights) catch {
        std.heap.page_allocator.free(pack_weights.data);
        return null;
    };
    const cached = pointwise_pack_cache.get(key).?;
    putThreadLocalPackedPointwiseWeights(key, cached);
    return cached;
}

fn buildPackedPointwiseWeights(
    weights: *const common.Tensor,
    groups: usize,
) ?PackedPointwiseWeights {
    const out_channels = weights.shape[0];
    const in_per_group = weights.shape[1];
    if (groups == 0 or out_channels % groups != 0) return null;
    const out_per_group = out_channels / groups;
    const total_weights = out_channels * in_per_group;

    const packed_data = std.heap.page_allocator.alloc(f32, total_weights) catch return null;
    errdefer std.heap.page_allocator.free(packed_data);

    for (0..groups) |group_idx| {
        for (0..in_per_group) |ic_local| {
            const packed_row_base = (group_idx * in_per_group + ic_local) * out_per_group;
            for (0..out_per_group) |oc_local| {
                const oc = group_idx * out_per_group + oc_local;
                packed_data[packed_row_base + oc_local] = weights.data[oc * in_per_group + ic_local];
            }
        }
    }

    return PackedPointwiseWeights{
        .out_channels = out_channels,
        .in_per_group = in_per_group,
        .groups = groups,
        .out_per_group = out_per_group,
        .data = packed_data,
    };
}

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
    const out_channels = weights.shape[0];
    const ctx = PointwiseContext{
        .input = input,
        .weights = weights,
        .bias = bias,
        .output = output,
        .groups = groups,
        .apply_silu = apply_silu,
    };
    return parallel.runByOutputChannel(
        PointwiseContext,
        &ctx,
        out_channels,
        thread_count,
        tasks.Conv2DPointwiseTask,
        makePointwiseTask,
        conv2dPointwiseWorker,
        runPointwiseRange,
    );
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
    const out_channels = weights.shape[0];
    const ctx = PointwiseConcatContext{
        .inputs = inputs,
        .input_channel_offsets = input_channel_offsets,
        .weights = weights,
        .bias = bias,
        .output = output,
        .apply_silu = apply_silu,
    };
    return parallel.runByOutputChannel(
        PointwiseConcatContext,
        &ctx,
        out_channels,
        thread_count,
        tasks.Conv2DPointwiseConcatTask,
        makePointwiseConcatTask,
        conv2dPointwiseConcatWorker,
        runPointwiseConcatRange,
    );
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
                const weight0_base = oc * in_per_group;
                const weight1_base = (oc + 1) * in_per_group;
                const weight2_base = (oc + 2) * in_per_group;
                const weight3_base = (oc + 3) * in_per_group;
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    var acc2 = @as(common.F32xN, @splat(bias2));
                    var acc3 = @as(common.F32xN, @splat(bias3));
                    for (0..in_per_group) |ic_local| {
                        const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        acc0 += src * @as(common.F32xN, @splat(weights.data[weight0_base + ic_local]));
                        acc1 += src * @as(common.F32xN, @splat(weights.data[weight1_base + ic_local]));
                        acc2 += src * @as(common.F32xN, @splat(weights.data[weight2_base + ic_local]));
                        acc3 += src * @as(common.F32xN, @splat(weights.data[weight3_base + ic_local]));
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                    common.storeF32xN(out2_slice, i, common.maybeApplySiluVector(acc2, apply_silu));
                    common.storeF32xN(out3_slice, i, common.maybeApplySiluVector(acc3, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    var acc2: f32 = bias2;
                    var acc3: f32 = bias3;
                    for (0..in_per_group) |ic_local| {
                        const src = input.data[input_batch_base + (in_channel_start + ic_local) * plane + i];
                        acc0 += src * weights.data[weight0_base + ic_local];
                        acc1 += src * weights.data[weight1_base + ic_local];
                        acc2 += src * weights.data[weight2_base + ic_local];
                        acc3 += src * weights.data[weight3_base + ic_local];
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                    out2_slice[i] = common.maybeApplySilu(acc2, apply_silu);
                    out3_slice[i] = common.maybeApplySilu(acc3, apply_silu);
                }
                oc += 4;
                continue;
            }

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                const weight0_base = oc * in_per_group;
                const weight1_base = (oc + 1) * in_per_group;
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    for (0..in_per_group) |ic_local| {
                        const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        acc0 += src * @as(common.F32xN, @splat(weights.data[weight0_base + ic_local]));
                        acc1 += src * @as(common.F32xN, @splat(weights.data[weight1_base + ic_local]));
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    for (0..in_per_group) |ic_local| {
                        const src = input.data[input_batch_base + (in_channel_start + ic_local) * plane + i];
                        acc0 += src * weights.data[weight0_base + ic_local];
                        acc1 += src * weights.data[weight1_base + ic_local];
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                }
                oc += 2;
                continue;
            }

            const out_slice = output.data[output_batch_base + oc * plane ..][0..plane];
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            const weight_base = oc * in_per_group;
            var i: usize = 0;
            while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                var acc = @as(common.F32xN, @splat(bias_value));
                for (0..in_per_group) |ic_local| {
                    const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                    const src = common.loadF32xN(input_slice, i);
                    acc += src * @as(common.F32xN, @splat(weights.data[weight_base + ic_local]));
                }
                common.storeF32xN(out_slice, i, common.maybeApplySiluVector(acc, apply_silu));
            }
            while (i < plane) : (i += 1) {
                var acc: f32 = bias_value;
                for (0..in_per_group) |ic_local| {
                    acc += input.data[input_batch_base + (in_channel_start + ic_local) * plane + i] * weights.data[weight_base + ic_local];
                }
                out_slice[i] = common.maybeApplySilu(acc, apply_silu);
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
    if (getPackedPointwiseWeights(weights, 1)) |pack_weights| {
        if (pack_weights.out_channels == weights.shape[0] and pack_weights.in_per_group == weights.shape[1]) {
            conv2dPointwiseConcatRangePacked(inputs, input_channel_offsets, bias, output, oc_start, oc_end, apply_silu, pack_weights);
            return;
        }
    }

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
                const weight0_base = oc * weights.shape[1];
                const weight1_base = (oc + 1) * weights.shape[1];
                const weight2_base = (oc + 2) * weights.shape[1];
                const weight3_base = (oc + 3) * weights.shape[1];
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    var acc2 = @as(common.F32xN, @splat(bias2));
                    var acc3 = @as(common.F32xN, @splat(bias3));
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        for (0..input.shape[1]) |ic| {
                            const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                            const weight_index = channel_offset + ic;
                            const src = common.loadF32xN(input_slice, i);
                            acc0 += src * @as(common.F32xN, @splat(weights.data[weight0_base + weight_index]));
                            acc1 += src * @as(common.F32xN, @splat(weights.data[weight1_base + weight_index]));
                            acc2 += src * @as(common.F32xN, @splat(weights.data[weight2_base + weight_index]));
                            acc3 += src * @as(common.F32xN, @splat(weights.data[weight3_base + weight_index]));
                        }
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                    common.storeF32xN(out2_slice, i, common.maybeApplySiluVector(acc2, apply_silu));
                    common.storeF32xN(out3_slice, i, common.maybeApplySiluVector(acc3, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    var acc2: f32 = bias2;
                    var acc3: f32 = bias3;
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        for (0..input.shape[1]) |ic| {
                            const src = input.data[input_batch_base + ic * plane + i];
                            const weight_index = channel_offset + ic;
                            acc0 += src * weights.data[weight0_base + weight_index];
                            acc1 += src * weights.data[weight1_base + weight_index];
                            acc2 += src * weights.data[weight2_base + weight_index];
                            acc3 += src * weights.data[weight3_base + weight_index];
                        }
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                    out2_slice[i] = common.maybeApplySilu(acc2, apply_silu);
                    out3_slice[i] = common.maybeApplySilu(acc3, apply_silu);
                }
                oc += 4;
                continue;
            }

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                const weight0_base = oc * weights.shape[1];
                const weight1_base = (oc + 1) * weights.shape[1];
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        for (0..input.shape[1]) |ic| {
                            const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                            const weight_index = channel_offset + ic;
                            const src = common.loadF32xN(input_slice, i);
                            acc0 += src * @as(common.F32xN, @splat(weights.data[weight0_base + weight_index]));
                            acc1 += src * @as(common.F32xN, @splat(weights.data[weight1_base + weight_index]));
                        }
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        for (0..input.shape[1]) |ic| {
                            const src = input.data[input_batch_base + ic * plane + i];
                            const weight_index = channel_offset + ic;
                            acc0 += src * weights.data[weight0_base + weight_index];
                            acc1 += src * weights.data[weight1_base + weight_index];
                        }
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                }
                oc += 2;
                continue;
            }

            const out_slice = output.data[output_batch_base + oc * plane ..][0..plane];
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            const weight_base = oc * weights.shape[1];
            var i: usize = 0;
            while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                var acc = @as(common.F32xN, @splat(bias_value));
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    for (0..input.shape[1]) |ic| {
                        const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        acc += src * @as(common.F32xN, @splat(weights.data[weight_base + channel_offset + ic]));
                    }
                }
                common.storeF32xN(out_slice, i, common.maybeApplySiluVector(acc, apply_silu));
            }
            while (i < plane) : (i += 1) {
                var acc: f32 = bias_value;
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    for (0..input.shape[1]) |ic| {
                        acc += input.data[input_batch_base + ic * plane + i] * weights.data[weight_base + channel_offset + ic];
                    }
                }
                out_slice[i] = common.maybeApplySilu(acc, apply_silu);
            }
            oc += 1;
        }
    }
}

fn conv2dPointwiseRangePacked(
    input: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
    pack_weights: PackedPointwiseWeights,
) void {
    const batch = input.shape[0];
    const in_channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];
    const out_channels = pack_weights.out_channels;
    const in_per_group = pack_weights.in_per_group;
    const out_per_group = pack_weights.out_per_group;
    const plane = height * width;

    for (0..batch) |n| {
        const input_batch_base = n * in_channels * plane;
        const output_batch_base = n * out_channels * plane;
        var oc = oc_start;
        while (oc < oc_end) {
            const group_idx = oc / out_per_group;
            const in_channel_start = group_idx * in_per_group;
            const oc_in_group = oc - group_idx * out_per_group;
            const quadable = oc + 3 < oc_end and
                (oc + 3) / out_per_group == group_idx;
            const pairable = oc + 1 < oc_end and
                (oc + 1) / out_per_group == group_idx;

            if (quadable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const out2_slice = output.data[output_batch_base + (oc + 2) * plane ..][0..plane];
                const out3_slice = output.data[output_batch_base + (oc + 3) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                const bias2: f32 = if (bias) |bias_values| bias_values[oc + 2] else 0.0;
                const bias3: f32 = if (bias) |bias_values| bias_values[oc + 3] else 0.0;
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    var acc2 = @as(common.F32xN, @splat(bias2));
                    var acc3 = @as(common.F32xN, @splat(bias3));
                    for (0..in_per_group) |ic_local| {
                        const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                        acc0 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                        acc1 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 1]));
                        acc2 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 2]));
                        acc3 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 3]));
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                    common.storeF32xN(out2_slice, i, common.maybeApplySiluVector(acc2, apply_silu));
                    common.storeF32xN(out3_slice, i, common.maybeApplySiluVector(acc3, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    var acc2: f32 = bias2;
                    var acc3: f32 = bias3;
                    for (0..in_per_group) |ic_local| {
                        const src = input.data[input_batch_base + (in_channel_start + ic_local) * plane + i];
                        const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                        acc0 += src * pack_weights.data[packed_base];
                        acc1 += src * pack_weights.data[packed_base + 1];
                        acc2 += src * pack_weights.data[packed_base + 2];
                        acc3 += src * pack_weights.data[packed_base + 3];
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                    out2_slice[i] = common.maybeApplySilu(acc2, apply_silu);
                    out3_slice[i] = common.maybeApplySilu(acc3, apply_silu);
                }
                oc += 4;
                continue;
            }

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    for (0..in_per_group) |ic_local| {
                        const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                        acc0 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                        acc1 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 1]));
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    for (0..in_per_group) |ic_local| {
                        const src = input.data[input_batch_base + (in_channel_start + ic_local) * plane + i];
                        const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                        acc0 += src * pack_weights.data[packed_base];
                        acc1 += src * pack_weights.data[packed_base + 1];
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                }
                oc += 2;
                continue;
            }

            const out_slice = output.data[output_batch_base + oc * plane ..][0..plane];
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            var i: usize = 0;
            while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                var acc = @as(common.F32xN, @splat(bias_value));
                for (0..in_per_group) |ic_local| {
                    const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
                    const src = common.loadF32xN(input_slice, i);
                    const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                    acc += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                }
                common.storeF32xN(out_slice, i, common.maybeApplySiluVector(acc, apply_silu));
            }
            while (i < plane) : (i += 1) {
                var acc: f32 = bias_value;
                for (0..in_per_group) |ic_local| {
                    const packed_base = (group_idx * in_per_group + ic_local) * out_per_group + oc_in_group;
                    acc += input.data[input_batch_base + (in_channel_start + ic_local) * plane + i] * pack_weights.data[packed_base];
                }
                out_slice[i] = common.maybeApplySilu(acc, apply_silu);
            }
            oc += 1;
        }
    }
}

fn conv2dPointwiseConcatRangePacked(
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    bias: ?[]const f32,
    output: *common.Tensor,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
    pack_weights: PackedPointwiseWeights,
) void {
    const batch = output.shape[0];
    const out_channels = pack_weights.out_channels;
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
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    var acc2 = @as(common.F32xN, @splat(bias2));
                    var acc3 = @as(common.F32xN, @splat(bias3));
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        var packed_base = channel_offset * out_channels + oc;
                        for (0..input.shape[1]) |ic| {
                            const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                            const src = common.loadF32xN(input_slice, i);
                            acc0 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                            acc1 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 1]));
                            acc2 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 2]));
                            acc3 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 3]));
                            packed_base += out_channels;
                        }
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                    common.storeF32xN(out2_slice, i, common.maybeApplySiluVector(acc2, apply_silu));
                    common.storeF32xN(out3_slice, i, common.maybeApplySiluVector(acc3, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    var acc2: f32 = bias2;
                    var acc3: f32 = bias3;
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        var packed_base = channel_offset * out_channels + oc;
                        for (0..input.shape[1]) |ic| {
                            const src = input.data[input_batch_base + ic * plane + i];
                            acc0 += src * pack_weights.data[packed_base];
                            acc1 += src * pack_weights.data[packed_base + 1];
                            acc2 += src * pack_weights.data[packed_base + 2];
                            acc3 += src * pack_weights.data[packed_base + 3];
                            packed_base += out_channels;
                        }
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                    out2_slice[i] = common.maybeApplySilu(acc2, apply_silu);
                    out3_slice[i] = common.maybeApplySilu(acc3, apply_silu);
                }
                oc += 4;
                continue;
            }

            if (pairable) {
                const out0_slice = output.data[output_batch_base + oc * plane ..][0..plane];
                const out1_slice = output.data[output_batch_base + (oc + 1) * plane ..][0..plane];
                const bias0: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
                const bias1: f32 = if (bias) |bias_values| bias_values[oc + 1] else 0.0;
                var i: usize = 0;
                while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                    var acc0 = @as(common.F32xN, @splat(bias0));
                    var acc1 = @as(common.F32xN, @splat(bias1));
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        var packed_base = channel_offset * out_channels + oc;
                        for (0..input.shape[1]) |ic| {
                            const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                            const src = common.loadF32xN(input_slice, i);
                            acc0 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                            acc1 += src * @as(common.F32xN, @splat(pack_weights.data[packed_base + 1]));
                            packed_base += out_channels;
                        }
                    }
                    common.storeF32xN(out0_slice, i, common.maybeApplySiluVector(acc0, apply_silu));
                    common.storeF32xN(out1_slice, i, common.maybeApplySiluVector(acc1, apply_silu));
                }
                while (i < plane) : (i += 1) {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    for (inputs, input_channel_offsets) |input, channel_offset| {
                        const input_batch_base = n * input.shape[1] * plane;
                        var packed_base = channel_offset * out_channels + oc;
                        for (0..input.shape[1]) |ic| {
                            const src = input.data[input_batch_base + ic * plane + i];
                            acc0 += src * pack_weights.data[packed_base];
                            acc1 += src * pack_weights.data[packed_base + 1];
                            packed_base += out_channels;
                        }
                    }
                    out0_slice[i] = common.maybeApplySilu(acc0, apply_silu);
                    out1_slice[i] = common.maybeApplySilu(acc1, apply_silu);
                }
                oc += 2;
                continue;
            }

            const out_slice = output.data[output_batch_base + oc * plane ..][0..plane];
            const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;
            var i: usize = 0;
            while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
                var acc = @as(common.F32xN, @splat(bias_value));
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    var packed_base = channel_offset * out_channels + oc;
                    for (0..input.shape[1]) |ic| {
                        const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                        const src = common.loadF32xN(input_slice, i);
                        acc += src * @as(common.F32xN, @splat(pack_weights.data[packed_base]));
                        packed_base += out_channels;
                    }
                }
                common.storeF32xN(out_slice, i, common.maybeApplySiluVector(acc, apply_silu));
            }
            while (i < plane) : (i += 1) {
                var acc: f32 = bias_value;
                for (inputs, input_channel_offsets) |input, channel_offset| {
                    const input_batch_base = n * input.shape[1] * plane;
                    var packed_base = channel_offset * out_channels + oc;
                    for (0..input.shape[1]) |ic| {
                        acc += input.data[input_batch_base + ic * plane + i] * pack_weights.data[packed_base];
                        packed_base += out_channels;
                    }
                }
                out_slice[i] = common.maybeApplySilu(acc, apply_silu);
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

const PointwiseContext = struct {
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    groups: usize,
    apply_silu: bool,
};

const PointwiseConcatContext = struct {
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    apply_silu: bool,
};

fn makePointwiseTask(ctx: *const PointwiseContext, oc_start: usize, oc_end: usize) tasks.Conv2DPointwiseTask {
    return .{
        .input = ctx.input,
        .weights = ctx.weights,
        .bias = ctx.bias,
        .output = ctx.output,
        .groups = ctx.groups,
        .oc_start = oc_start,
        .oc_end = oc_end,
        .apply_silu = ctx.apply_silu,
    };
}

fn makePointwiseConcatTask(ctx: *const PointwiseConcatContext, oc_start: usize, oc_end: usize) tasks.Conv2DPointwiseConcatTask {
    return .{
        .inputs = ctx.inputs,
        .input_channel_offsets = ctx.input_channel_offsets,
        .weights = ctx.weights,
        .bias = ctx.bias,
        .output = ctx.output,
        .oc_start = oc_start,
        .oc_end = oc_end,
        .apply_silu = ctx.apply_silu,
    };
}

fn runPointwiseRange(ctx: *const PointwiseContext, oc_start: usize, oc_end: usize) common.OpError!void {
    return conv2dPointwiseRange(ctx.input, ctx.weights, ctx.bias, ctx.output, ctx.groups, oc_start, oc_end, ctx.apply_silu);
}

fn runPointwiseConcatRange(ctx: *const PointwiseConcatContext, oc_start: usize, oc_end: usize) common.OpError!void {
    return conv2dPointwiseConcatRange(
        ctx.inputs,
        ctx.input_channel_offsets,
        ctx.weights,
        ctx.bias,
        ctx.output,
        oc_start,
        oc_end,
        ctx.apply_silu,
    );
}
