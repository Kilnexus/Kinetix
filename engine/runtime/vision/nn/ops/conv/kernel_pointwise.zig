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

// Cache
const PointwisePackCache = struct {
    const threadlocal_capacity = 16;

    var mutex: std.Thread.Mutex = .{};
    var cache: std.AutoHashMapUnmanaged(PointwisePackKey, PackedPointwiseWeights) = .{};
    threadlocal var tl_len: usize = 0;
    threadlocal var tl_next: usize = 0;
    threadlocal var tl_keys: [threadlocal_capacity]PointwisePackKey = undefined;
    threadlocal var tl_values: [threadlocal_capacity]PackedPointwiseWeights = undefined;

    fn get(weights: *const common.Tensor, groups: usize) ?PackedPointwiseWeights {
        const key = makeKey(weights, groups) orelse return null;
        if (getThreadLocal(key)) |cached| return cached;

        mutex.lock();
        defer mutex.unlock();

        if (cache.get(key)) |cached| {
            putThreadLocal(key, cached);
            return cached;
        }

        const pack_weights = build(weights, groups) orelse return null;
        cache.put(std.heap.page_allocator, key, pack_weights) catch {
            std.heap.page_allocator.free(pack_weights.data);
            return null;
        };
        const cached = cache.get(key).?;
        putThreadLocal(key, cached);
        return cached;
    }

    fn makeKey(weights: *const common.Tensor, groups: usize) ?PointwisePackKey {
        if (groups == 0) return null;
        const out_channels = weights.shape[0];
        const in_per_group = weights.shape[1];
        if (out_channels == 0 or in_per_group == 0) return null;
        if (out_channels % groups != 0) return null;

        return .{
            .ptr = @intFromPtr(weights.data.ptr),
            .out_channels = out_channels,
            .in_per_group = in_per_group,
            .groups = groups,
        };
    }

    fn keyEqual(lhs: PointwisePackKey, rhs: PointwisePackKey) bool {
        return lhs.ptr == rhs.ptr and
            lhs.out_channels == rhs.out_channels and
            lhs.in_per_group == rhs.in_per_group and
            lhs.groups == rhs.groups;
    }

    fn getThreadLocal(key: PointwisePackKey) ?PackedPointwiseWeights {
        for (0..tl_len) |index| {
            if (keyEqual(tl_keys[index], key)) {
                return tl_values[index];
            }
        }
        return null;
    }

    fn putThreadLocal(key: PointwisePackKey, pack_weights: PackedPointwiseWeights) void {
        for (0..tl_len) |index| {
            if (keyEqual(tl_keys[index], key)) {
                tl_values[index] = pack_weights;
                return;
            }
        }

        if (tl_len < threadlocal_capacity) {
            const index = tl_len;
            tl_keys[index] = key;
            tl_values[index] = pack_weights;
            tl_len += 1;
            return;
        }

        const replace_index = tl_next;
        tl_keys[replace_index] = key;
        tl_values[replace_index] = pack_weights;
        tl_next = (tl_next + 1) % threadlocal_capacity;
    }

    fn build(weights: *const common.Tensor, groups: usize) ?PackedPointwiseWeights {
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

        return .{
            .out_channels = out_channels,
            .in_per_group = in_per_group,
            .groups = groups,
            .out_per_group = out_per_group,
            .data = packed_data,
        };
    }
};

// Public entry points
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

// Parallel entry points
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

// Parallel callbacks
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

// Range implementations
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
    if (PointwisePackCache.get(weights, groups)) |pack_weights| {
        if (pack_weights.out_channels == weights.shape[0] and
            pack_weights.in_per_group == weights.shape[1] and
            pack_weights.groups == groups)
        {
            conv2dPointwiseRangePacked(input, bias, output, oc_start, oc_end, apply_silu, pack_weights);
            return;
        }
    }

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
                runPointwiseBlock(
                    4,
                    DensePointwiseAccessor{
                        .weights = weights,
                        .lane_base = oc,
                        .in_per_group = in_per_group,
                    },
                    input,
                    output,
                    input_batch_base,
                    output_batch_base,
                    in_channel_start,
                    in_per_group,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                );
                oc += 4;
                continue;
            }

            if (pairable) {
                runPointwiseBlock(
                    2,
                    DensePointwiseAccessor{
                        .weights = weights,
                        .lane_base = oc,
                        .in_per_group = in_per_group,
                    },
                    input,
                    output,
                    input_batch_base,
                    output_batch_base,
                    in_channel_start,
                    in_per_group,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                );
                oc += 2;
                continue;
            }

            runPointwiseBlock(
                1,
                DensePointwiseAccessor{
                    .weights = weights,
                    .lane_base = oc,
                    .in_per_group = in_per_group,
                },
                input,
                output,
                input_batch_base,
                output_batch_base,
                in_channel_start,
                in_per_group,
                plane,
                oc,
                bias,
                apply_silu,
            );
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
    if (PointwisePackCache.get(weights, 1)) |pack_weights| {
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
                runPointwiseConcatBlock(
                    4,
                    DensePointwiseConcatAccessor{
                        .weights = weights,
                        .lane_base = oc,
                    },
                    inputs,
                    input_channel_offsets,
                    output,
                    output_batch_base,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                    n,
                );
                oc += 4;
                continue;
            }

            if (pairable) {
                runPointwiseConcatBlock(
                    2,
                    DensePointwiseConcatAccessor{
                        .weights = weights,
                        .lane_base = oc,
                    },
                    inputs,
                    input_channel_offsets,
                    output,
                    output_batch_base,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                    n,
                );
                oc += 2;
                continue;
            }

            runPointwiseConcatBlock(
                1,
                DensePointwiseConcatAccessor{
                    .weights = weights,
                    .lane_base = oc,
                },
                inputs,
                input_channel_offsets,
                output,
                output_batch_base,
                plane,
                oc,
                bias,
                apply_silu,
                n,
            );
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
                runPointwiseBlock(
                    4,
                    PackedPointwiseAccessor{
                        .data = pack_weights.data,
                        .group_idx = group_idx,
                        .in_per_group = in_per_group,
                        .out_per_group = out_per_group,
                        .oc_in_group = oc_in_group,
                    },
                    input,
                    output,
                    input_batch_base,
                    output_batch_base,
                    in_channel_start,
                    in_per_group,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                );
                oc += 4;
                continue;
            }

            if (pairable) {
                runPointwiseBlock(
                    2,
                    PackedPointwiseAccessor{
                        .data = pack_weights.data,
                        .group_idx = group_idx,
                        .in_per_group = in_per_group,
                        .out_per_group = out_per_group,
                        .oc_in_group = oc_in_group,
                    },
                    input,
                    output,
                    input_batch_base,
                    output_batch_base,
                    in_channel_start,
                    in_per_group,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                );
                oc += 2;
                continue;
            }

            runPointwiseBlock(
                1,
                PackedPointwiseAccessor{
                    .data = pack_weights.data,
                    .group_idx = group_idx,
                    .in_per_group = in_per_group,
                    .out_per_group = out_per_group,
                    .oc_in_group = oc_in_group,
                },
                input,
                output,
                input_batch_base,
                output_batch_base,
                in_channel_start,
                in_per_group,
                plane,
                oc,
                bias,
                apply_silu,
            );
            oc += 1;
        }
    }
}

// Block helpers
const DensePointwiseAccessor = struct {
    weights: *const common.Tensor,
    lane_base: usize,
    in_per_group: usize,

    inline fn weight(self: @This(), ic_local: usize, lane: usize) f32 {
        return self.weights.data[(self.lane_base + lane) * self.in_per_group + ic_local];
    }
};

const PackedPointwiseAccessor = struct {
    data: []const f32,
    group_idx: usize,
    in_per_group: usize,
    out_per_group: usize,
    oc_in_group: usize,

    inline fn weight(self: @This(), ic_local: usize, lane: usize) f32 {
        const base = (self.group_idx * self.in_per_group + ic_local) * self.out_per_group + self.oc_in_group;
        return self.data[base + lane];
    }
};

fn runPointwiseBlock(
    comptime lanes: usize,
    accessor: anytype,
    input: *const common.Tensor,
    output: *common.Tensor,
    input_batch_base: usize,
    output_batch_base: usize,
    in_channel_start: usize,
    in_per_group: usize,
    plane: usize,
    oc: usize,
    bias: ?[]const f32,
    apply_silu: bool,
) void {
    var out_slices: [lanes][]f32 = undefined;
    var bias_values: [lanes]f32 = undefined;
    inline for (0..lanes) |lane| {
        out_slices[lane] = output.data[output_batch_base + (oc + lane) * plane ..][0..plane];
        bias_values[lane] = if (bias) |bias_values_slice| bias_values_slice[oc + lane] else 0.0;
    }

    var i: usize = 0;
    while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
        var acc: [lanes]common.F32xN = undefined;
        inline for (0..lanes) |lane| {
            acc[lane] = @as(common.F32xN, @splat(bias_values[lane]));
        }

        for (0..in_per_group) |ic_local| {
            const input_slice = input.data[input_batch_base + (in_channel_start + ic_local) * plane ..][0..plane];
            const src = common.loadF32xN(input_slice, i);
            inline for (0..lanes) |lane| {
                acc[lane] += src * @as(common.F32xN, @splat(accessor.weight(ic_local, lane)));
            }
        }

        inline for (0..lanes) |lane| {
            common.storeF32xN(out_slices[lane], i, common.maybeApplySiluVector(acc[lane], apply_silu));
        }
    }

    while (i < plane) : (i += 1) {
        var acc: [lanes]f32 = undefined;
        inline for (0..lanes) |lane| {
            acc[lane] = bias_values[lane];
        }

        for (0..in_per_group) |ic_local| {
            const src = input.data[input_batch_base + (in_channel_start + ic_local) * plane + i];
            inline for (0..lanes) |lane| {
                acc[lane] += src * accessor.weight(ic_local, lane);
            }
        }

        inline for (0..lanes) |lane| {
            out_slices[lane][i] = common.maybeApplySilu(acc[lane], apply_silu);
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
                runPointwiseConcatBlock(
                    4,
                    PackedPointwiseConcatAccessor{
                        .data = pack_weights.data,
                        .out_channels = out_channels,
                        .lane_base = oc,
                    },
                    inputs,
                    input_channel_offsets,
                    output,
                    output_batch_base,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                    n,
                );
                oc += 4;
                continue;
            }

            if (pairable) {
                runPointwiseConcatBlock(
                    2,
                    PackedPointwiseConcatAccessor{
                        .data = pack_weights.data,
                        .out_channels = out_channels,
                        .lane_base = oc,
                    },
                    inputs,
                    input_channel_offsets,
                    output,
                    output_batch_base,
                    plane,
                    oc,
                    bias,
                    apply_silu,
                    n,
                );
                oc += 2;
                continue;
            }

            runPointwiseConcatBlock(
                1,
                PackedPointwiseConcatAccessor{
                    .data = pack_weights.data,
                    .out_channels = out_channels,
                    .lane_base = oc,
                },
                inputs,
                input_channel_offsets,
                output,
                output_batch_base,
                plane,
                oc,
                bias,
                apply_silu,
                n,
            );
            oc += 1;
        }
    }
}

const DensePointwiseConcatAccessor = struct {
    weights: *const common.Tensor,
    lane_base: usize,

    inline fn weight(self: @This(), channel_index: usize, lane: usize) f32 {
        return self.weights.data[(self.lane_base + lane) * self.weights.shape[1] + channel_index];
    }
};

const PackedPointwiseConcatAccessor = struct {
    data: []const f32,
    out_channels: usize,
    lane_base: usize,

    inline fn weight(self: @This(), channel_index: usize, lane: usize) f32 {
        return self.data[channel_index * self.out_channels + self.lane_base + lane];
    }
};

fn runPointwiseConcatBlock(
    comptime lanes: usize,
    accessor: anytype,
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    output: *common.Tensor,
    output_batch_base: usize,
    plane: usize,
    oc: usize,
    bias: ?[]const f32,
    apply_silu: bool,
    batch_index: usize,
) void {
    var out_slices: [lanes][]f32 = undefined;
    var bias_values: [lanes]f32 = undefined;
    inline for (0..lanes) |lane| {
        out_slices[lane] = output.data[output_batch_base + (oc + lane) * plane ..][0..plane];
        bias_values[lane] = if (bias) |bias_values_slice| bias_values_slice[oc + lane] else 0.0;
    }

    var i: usize = 0;
    while (i + common.simd_lane_count <= plane) : (i += common.simd_lane_count) {
        var acc: [lanes]common.F32xN = undefined;
        inline for (0..lanes) |lane| {
            acc[lane] = @as(common.F32xN, @splat(bias_values[lane]));
        }

        for (inputs, input_channel_offsets) |input, channel_offset| {
            const input_batch_base = batch_index * input.shape[1] * plane;
            for (0..input.shape[1]) |ic| {
                const input_slice = input.data[input_batch_base + ic * plane ..][0..plane];
                const src = common.loadF32xN(input_slice, i);
                const channel_index = channel_offset + ic;
                inline for (0..lanes) |lane| {
                    acc[lane] += src * @as(common.F32xN, @splat(accessor.weight(channel_index, lane)));
                }
            }
        }

        inline for (0..lanes) |lane| {
            common.storeF32xN(out_slices[lane], i, common.maybeApplySiluVector(acc[lane], apply_silu));
        }
    }

    while (i < plane) : (i += 1) {
        var acc: [lanes]f32 = undefined;
        inline for (0..lanes) |lane| {
            acc[lane] = bias_values[lane];
        }

        for (inputs, input_channel_offsets) |input, channel_offset| {
            const input_batch_base = batch_index * input.shape[1] * plane;
            for (0..input.shape[1]) |ic| {
                const channel_index = channel_offset + ic;
                const src = input.data[input_batch_base + ic * plane + i];
                inline for (0..lanes) |lane| {
                    acc[lane] += src * accessor.weight(channel_index, lane);
                }
            }
        }

        inline for (0..lanes) |lane| {
            out_slices[lane][i] = common.maybeApplySilu(acc[lane], apply_silu);
        }
    }
}
