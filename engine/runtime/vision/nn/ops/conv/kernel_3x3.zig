const std = @import("std");
const common = @import("common.zig");
const tasks = @import("tasks.zig");
const thread_pool = @import("engine_global_thread_pool");

pub fn conv2d3x3Pad1(
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

    if (options.stride_h == 2 and options.stride_w == 2) {
        if (thread_count > 1) {
            return runParallelByOutputChannel(Stride2, input, weights, bias, output, options, thread_count);
        }
        return Stride2.range(input, weights, bias, output, options, 0, out_channels);
    }

    if (thread_count > 1) {
        return runParallelByOutputChannel(Stride1, input, weights, bias, output, options, thread_count);
    }
    return Stride1.range(input, weights, bias, output, options, 0, out_channels);
}

fn runParallelByOutputChannel(
    comptime Impl: type,
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    thread_count: usize,
) common.OpError!void {
    if (thread_pool.get()) |pool| {
        const out_channels = weights.shape[0];
        var wg: std.Thread.WaitGroup = .{};
        for (0..thread_count) |thread_index| {
            const oc_start = (out_channels * thread_index) / thread_count;
            const oc_end = (out_channels * (thread_index + 1)) / thread_count;
            if (oc_start == oc_end) continue;

            const task = tasks.Conv2DTask{
                .input = input,
                .weights = weights,
                .bias = bias,
                .output = output,
                .options = options,
                .oc_start = oc_start,
                .oc_end = oc_end,
            };
            if (thread_index + 1 == thread_count) {
                Impl.worker(task);
            } else {
                pool.spawnWg(&wg, Impl.worker, .{task});
            }
        }
        wg.wait();
        return;
    }

    var threads: [common.max_supported_conv_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const out_channels = weights.shape[0];

    for (0..thread_count) |thread_index| {
        const oc_start = (out_channels * thread_index) / thread_count;
        const oc_end = (out_channels * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            try Impl.range(input, weights, bias, output, options, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, Impl.worker, .{
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
                try Impl.range(input, weights, bias, output, options, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
}

const Stride1 = struct {
    fn range(
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
        const can_vectorize_width = expected_w > common.simd_lane_count + 1;

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
                    const interior_row = oy > 0 and oy + 1 < expected_h;

                    if (!interior_row or !can_vectorize_width) {
                        for (0..expected_w) |ox| {
                            const point_value = pointPair(
                                input,
                                weights,
                                input_batch_base,
                                weights0_channel_base,
                                weights1_channel_base,
                                oy,
                                ox,
                                bias0,
                                bias1,
                            );
                            output.data[output0_row_base + ox] = common.maybeApplySilu(point_value[0], options.apply_silu);
                            output.data[output1_row_base + ox] = common.maybeApplySilu(point_value[1], options.apply_silu);
                        }
                        continue;
                    }

                    const first = pointPair(
                        input,
                        weights,
                        input_batch_base,
                        weights0_channel_base,
                        weights1_channel_base,
                        oy,
                        0,
                        bias0,
                        bias1,
                    );
                    output.data[output0_row_base] = common.maybeApplySilu(first[0], options.apply_silu);
                    output.data[output1_row_base] = common.maybeApplySilu(first[1], options.apply_silu);

                    var ox: usize = 1;
                    const interior_w_end = expected_w - 1;
                    while (ox + common.simd_lane_count <= interior_w_end) : (ox += common.simd_lane_count) {
                        const acc = interiorVecPair(
                            input,
                            weights,
                            input_batch_base,
                            weights0_channel_base,
                            weights1_channel_base,
                            oy,
                            ox,
                            bias0,
                            bias1,
                        );
                        common.storeF32xN(
                            output.data[output0_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc[0], options.apply_silu),
                        );
                        common.storeF32xN(
                            output.data[output1_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc[1], options.apply_silu),
                        );
                    }

                    while (ox < interior_w_end) : (ox += 1) {
                        const point_value = pointPair(
                            input,
                            weights,
                            input_batch_base,
                            weights0_channel_base,
                            weights1_channel_base,
                            oy,
                            ox,
                            bias0,
                            bias1,
                        );
                        output.data[output0_row_base + ox] = common.maybeApplySilu(point_value[0], options.apply_silu);
                        output.data[output1_row_base + ox] = common.maybeApplySilu(point_value[1], options.apply_silu);
                    }

                    const last = pointPair(
                        input,
                        weights,
                        input_batch_base,
                        weights0_channel_base,
                        weights1_channel_base,
                        oy,
                        expected_w - 1,
                        bias0,
                        bias1,
                    );
                    output.data[output0_row_base + expected_w - 1] = common.maybeApplySilu(last[0], options.apply_silu);
                    output.data[output1_row_base + expected_w - 1] = common.maybeApplySilu(last[1], options.apply_silu);
                }
            }

            while (oc < oc_end) : (oc += 1) {
                const weights_channel_base = oc * in_channels * 9;
                const output_channel_base = output_batch_base + oc * output_plane;
                const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;

                for (0..expected_h) |oy| {
                    const output_row_base = output_channel_base + oy * expected_w;
                    const interior_row = oy > 0 and oy + 1 < expected_h;

                    if (!interior_row or !can_vectorize_width) {
                        for (0..expected_w) |ox| {
                            output.data[output_row_base + ox] = common.maybeApplySilu(
                                point(
                                    input,
                                    weights,
                                    input_batch_base,
                                    weights_channel_base,
                                    oy,
                                    ox,
                                    bias_value,
                                ),
                                options.apply_silu,
                            );
                        }
                        continue;
                    }

                    output.data[output_row_base] = common.maybeApplySilu(
                        point(
                            input,
                            weights,
                            input_batch_base,
                            weights_channel_base,
                            oy,
                            0,
                            bias_value,
                        ),
                        options.apply_silu,
                    );

                    var ox: usize = 1;
                    const interior_w_end = expected_w - 1;
                    while (ox + common.simd_lane_count <= interior_w_end) : (ox += common.simd_lane_count) {
                        const acc = interiorVec(
                            input,
                            weights,
                            input_batch_base,
                            weights_channel_base,
                            oy,
                            ox,
                            bias_value,
                        );
                        common.storeF32xN(
                            output.data[output_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc, options.apply_silu),
                        );
                    }

                    while (ox < interior_w_end) : (ox += 1) {
                        output.data[output_row_base + ox] = common.maybeApplySilu(
                            point(
                                input,
                                weights,
                                input_batch_base,
                                weights_channel_base,
                                oy,
                                ox,
                                bias_value,
                            ),
                            options.apply_silu,
                        );
                    }

                    output.data[output_row_base + expected_w - 1] = common.maybeApplySilu(
                        point(
                            input,
                            weights,
                            input_batch_base,
                            weights_channel_base,
                            oy,
                            expected_w - 1,
                            bias_value,
                        ),
                        options.apply_silu,
                    );
                }
            }
        }
    }

    inline fn interiorVecPair(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights0_channel_base: usize,
        weights1_channel_base: usize,
        oy: usize,
        ox: usize,
        bias0: f32,
        bias1: f32,
    ) [2]common.F32xN {
        const in_channels = input.shape[1];
        const in_width = input.shape[3];
        const input_plane = input.shape[2] * input.shape[3];
        const row0 = (oy - 1) * in_width + (ox - 1);
        const row1 = row0 + in_width;
        const row2 = row1 + in_width;

        var acc0 = @as(common.F32xN, @splat(bias0));
        var acc1 = @as(common.F32xN, @splat(bias1));

        for (0..in_channels) |ic| {
            const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
            const weight0_base = weights0_channel_base + ic * 9;
            const weight1_base = weights1_channel_base + ic * 9;
            const r00 = common.loadF32xN(input_channel, row0);
            const r01 = common.loadF32xN(input_channel, row0 + 1);
            const r02 = common.loadF32xN(input_channel, row0 + 2);
            const r10 = common.loadF32xN(input_channel, row1);
            const r11 = common.loadF32xN(input_channel, row1 + 1);
            const r12 = common.loadF32xN(input_channel, row1 + 2);
            const r20 = common.loadF32xN(input_channel, row2);
            const r21 = common.loadF32xN(input_channel, row2 + 1);
            const r22 = common.loadF32xN(input_channel, row2 + 2);

            acc0 += r00 * @as(common.F32xN, @splat(weights.data[weight0_base + 0]));
            acc0 += r01 * @as(common.F32xN, @splat(weights.data[weight0_base + 1]));
            acc0 += r02 * @as(common.F32xN, @splat(weights.data[weight0_base + 2]));
            acc0 += r10 * @as(common.F32xN, @splat(weights.data[weight0_base + 3]));
            acc0 += r11 * @as(common.F32xN, @splat(weights.data[weight0_base + 4]));
            acc0 += r12 * @as(common.F32xN, @splat(weights.data[weight0_base + 5]));
            acc0 += r20 * @as(common.F32xN, @splat(weights.data[weight0_base + 6]));
            acc0 += r21 * @as(common.F32xN, @splat(weights.data[weight0_base + 7]));
            acc0 += r22 * @as(common.F32xN, @splat(weights.data[weight0_base + 8]));

            acc1 += r00 * @as(common.F32xN, @splat(weights.data[weight1_base + 0]));
            acc1 += r01 * @as(common.F32xN, @splat(weights.data[weight1_base + 1]));
            acc1 += r02 * @as(common.F32xN, @splat(weights.data[weight1_base + 2]));
            acc1 += r10 * @as(common.F32xN, @splat(weights.data[weight1_base + 3]));
            acc1 += r11 * @as(common.F32xN, @splat(weights.data[weight1_base + 4]));
            acc1 += r12 * @as(common.F32xN, @splat(weights.data[weight1_base + 5]));
            acc1 += r20 * @as(common.F32xN, @splat(weights.data[weight1_base + 6]));
            acc1 += r21 * @as(common.F32xN, @splat(weights.data[weight1_base + 7]));
            acc1 += r22 * @as(common.F32xN, @splat(weights.data[weight1_base + 8]));
        }

        return .{ acc0, acc1 };
    }

    inline fn interiorVec(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights_channel_base: usize,
        oy: usize,
        ox: usize,
        bias_value: f32,
    ) common.F32xN {
        const in_channels = input.shape[1];
        const in_width = input.shape[3];
        const input_plane = input.shape[2] * input.shape[3];
        const row0 = (oy - 1) * in_width + (ox - 1);
        const row1 = row0 + in_width;
        const row2 = row1 + in_width;

        var acc = @as(common.F32xN, @splat(bias_value));

        for (0..in_channels) |ic| {
            const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
            const weight_base = weights_channel_base + ic * 9;
            acc += common.loadF32xN(input_channel, row0) * @as(common.F32xN, @splat(weights.data[weight_base + 0]));
            acc += common.loadF32xN(input_channel, row0 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 1]));
            acc += common.loadF32xN(input_channel, row0 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 2]));
            acc += common.loadF32xN(input_channel, row1) * @as(common.F32xN, @splat(weights.data[weight_base + 3]));
            acc += common.loadF32xN(input_channel, row1 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 4]));
            acc += common.loadF32xN(input_channel, row1 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 5]));
            acc += common.loadF32xN(input_channel, row2) * @as(common.F32xN, @splat(weights.data[weight_base + 6]));
            acc += common.loadF32xN(input_channel, row2 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 7]));
            acc += common.loadF32xN(input_channel, row2 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 8]));
        }

        return acc;
    }

    inline fn pointPair(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights0_channel_base: usize,
        weights1_channel_base: usize,
        oy: usize,
        ox: usize,
        bias0: f32,
        bias1: f32,
    ) [2]f32 {
        return .{
            point(input, weights, input_batch_base, weights0_channel_base, oy, ox, bias0),
            point(input, weights, input_batch_base, weights1_channel_base, oy, ox, bias1),
        };
    }

    inline fn point(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights_channel_base: usize,
        oy: usize,
        ox: usize,
        bias_value: f32,
    ) f32 {
        const in_channels = input.shape[1];
        const in_height = input.shape[2];
        const in_width = input.shape[3];
        const input_plane = in_height * in_width;
        const base_y = @as(isize, @intCast(oy)) - 1;
        const base_x = @as(isize, @intCast(ox)) - 1;

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

    fn worker(task: tasks.Conv2DTask) void {
        range(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
    }
};

const Stride2 = struct {
    fn range(
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
        const can_vectorize_width = interior_w_end > common.simd_lane_count;

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
                    const interior_row = oy > 0 and oy < interior_h_end;

                    if (!interior_row or !can_vectorize_width) {
                        for (0..expected_w) |ox| {
                            const point_value = pointPair(
                                input,
                                weights,
                                input_batch_base,
                                weights0_channel_base,
                                weights1_channel_base,
                                oy,
                                ox,
                                bias0,
                                bias1,
                            );
                            output.data[output0_row_base + ox] = common.maybeApplySilu(point_value[0], options.apply_silu);
                            output.data[output1_row_base + ox] = common.maybeApplySilu(point_value[1], options.apply_silu);
                        }
                        continue;
                    }

                    const first = pointPair(
                        input,
                        weights,
                        input_batch_base,
                        weights0_channel_base,
                        weights1_channel_base,
                        oy,
                        0,
                        bias0,
                        bias1,
                    );
                    output.data[output0_row_base] = common.maybeApplySilu(first[0], options.apply_silu);
                    output.data[output1_row_base] = common.maybeApplySilu(first[1], options.apply_silu);

                    var ox: usize = 1;
                    while (ox + common.simd_lane_count <= interior_w_end) : (ox += common.simd_lane_count) {
                        const acc = interiorVecPair(
                            input,
                            weights,
                            input_batch_base,
                            weights0_channel_base,
                            weights1_channel_base,
                            oy,
                            ox,
                            bias0,
                            bias1,
                        );
                        common.storeF32xN(
                            output.data[output0_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc[0], options.apply_silu),
                        );
                        common.storeF32xN(
                            output.data[output1_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc[1], options.apply_silu),
                        );
                    }

                    while (ox < interior_w_end) : (ox += 1) {
                        const point_value = pointPair(
                            input,
                            weights,
                            input_batch_base,
                            weights0_channel_base,
                            weights1_channel_base,
                            oy,
                            ox,
                            bias0,
                            bias1,
                        );
                        output.data[output0_row_base + ox] = common.maybeApplySilu(point_value[0], options.apply_silu);
                        output.data[output1_row_base + ox] = common.maybeApplySilu(point_value[1], options.apply_silu);
                    }

                    for (interior_w_end..expected_w) |tail_ox| {
                        const point_value = pointPair(
                            input,
                            weights,
                            input_batch_base,
                            weights0_channel_base,
                            weights1_channel_base,
                            oy,
                            tail_ox,
                            bias0,
                            bias1,
                        );
                        output.data[output0_row_base + tail_ox] = common.maybeApplySilu(point_value[0], options.apply_silu);
                        output.data[output1_row_base + tail_ox] = common.maybeApplySilu(point_value[1], options.apply_silu);
                    }
                }
            }

            while (oc < oc_end) : (oc += 1) {
                const weights_channel_base = oc * in_channels * 9;
                const output_channel_base = output_batch_base + oc * output_plane;
                const bias_value: f32 = if (bias) |bias_values| bias_values[oc] else 0.0;

                for (0..expected_h) |oy| {
                    const output_row_base = output_channel_base + oy * expected_w;
                    const interior_row = oy > 0 and oy < interior_h_end;

                    if (!interior_row or !can_vectorize_width) {
                        for (0..expected_w) |ox| {
                            output.data[output_row_base + ox] = common.maybeApplySilu(
                                point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, ox),
                                options.apply_silu,
                            );
                        }
                        continue;
                    }

                    output.data[output_row_base] = common.maybeApplySilu(
                        point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, 0),
                        options.apply_silu,
                    );

                    var ox: usize = 1;
                    while (ox + common.simd_lane_count <= interior_w_end) : (ox += common.simd_lane_count) {
                        const acc = interiorVec(
                            input,
                            weights,
                            input_batch_base,
                            weights_channel_base,
                            oy,
                            ox,
                            bias_value,
                        );
                        common.storeF32xN(
                            output.data[output_row_base..][0..expected_w],
                            ox,
                            common.maybeApplySiluVector(acc, options.apply_silu),
                        );
                    }

                    while (ox < interior_w_end) : (ox += 1) {
                        output.data[output_row_base + ox] = common.maybeApplySilu(
                            point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, ox),
                            options.apply_silu,
                        );
                    }

                    for (interior_w_end..expected_w) |tail_ox| {
                        output.data[output_row_base + tail_ox] = common.maybeApplySilu(
                            point(input, weights, bias_value, input_batch_base, weights_channel_base, oy, tail_ox),
                            options.apply_silu,
                        );
                    }
                }
            }
        }
    }

    inline fn interiorVecPair(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights0_channel_base: usize,
        weights1_channel_base: usize,
        oy: usize,
        ox: usize,
        bias0: f32,
        bias1: f32,
    ) [2]common.F32xN {
        const in_channels = input.shape[1];
        const in_width = input.shape[3];
        const input_plane = input.shape[2] * input.shape[3];
        const row0 = (oy * 2 - 1) * in_width + (ox * 2 - 1);
        const row1 = row0 + in_width;
        const row2 = row1 + in_width;

        var acc0 = @as(common.F32xN, @splat(bias0));
        var acc1 = @as(common.F32xN, @splat(bias1));

        for (0..in_channels) |ic| {
            const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
            const weight0_base = weights0_channel_base + ic * 9;
            const weight1_base = weights1_channel_base + ic * 9;
            const r00 = loadF32xN(input_channel, row0);
            const r01 = loadF32xN(input_channel, row0 + 1);
            const r02 = loadF32xN(input_channel, row0 + 2);
            const r10 = loadF32xN(input_channel, row1);
            const r11 = loadF32xN(input_channel, row1 + 1);
            const r12 = loadF32xN(input_channel, row1 + 2);
            const r20 = loadF32xN(input_channel, row2);
            const r21 = loadF32xN(input_channel, row2 + 1);
            const r22 = loadF32xN(input_channel, row2 + 2);

            acc0 += r00 * @as(common.F32xN, @splat(weights.data[weight0_base + 0]));
            acc0 += r01 * @as(common.F32xN, @splat(weights.data[weight0_base + 1]));
            acc0 += r02 * @as(common.F32xN, @splat(weights.data[weight0_base + 2]));
            acc0 += r10 * @as(common.F32xN, @splat(weights.data[weight0_base + 3]));
            acc0 += r11 * @as(common.F32xN, @splat(weights.data[weight0_base + 4]));
            acc0 += r12 * @as(common.F32xN, @splat(weights.data[weight0_base + 5]));
            acc0 += r20 * @as(common.F32xN, @splat(weights.data[weight0_base + 6]));
            acc0 += r21 * @as(common.F32xN, @splat(weights.data[weight0_base + 7]));
            acc0 += r22 * @as(common.F32xN, @splat(weights.data[weight0_base + 8]));

            acc1 += r00 * @as(common.F32xN, @splat(weights.data[weight1_base + 0]));
            acc1 += r01 * @as(common.F32xN, @splat(weights.data[weight1_base + 1]));
            acc1 += r02 * @as(common.F32xN, @splat(weights.data[weight1_base + 2]));
            acc1 += r10 * @as(common.F32xN, @splat(weights.data[weight1_base + 3]));
            acc1 += r11 * @as(common.F32xN, @splat(weights.data[weight1_base + 4]));
            acc1 += r12 * @as(common.F32xN, @splat(weights.data[weight1_base + 5]));
            acc1 += r20 * @as(common.F32xN, @splat(weights.data[weight1_base + 6]));
            acc1 += r21 * @as(common.F32xN, @splat(weights.data[weight1_base + 7]));
            acc1 += r22 * @as(common.F32xN, @splat(weights.data[weight1_base + 8]));
        }

        return .{ acc0, acc1 };
    }

    inline fn interiorVec(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights_channel_base: usize,
        oy: usize,
        ox: usize,
        bias_value: f32,
    ) common.F32xN {
        const in_channels = input.shape[1];
        const in_width = input.shape[3];
        const input_plane = input.shape[2] * input.shape[3];
        const row0 = (oy * 2 - 1) * in_width + (ox * 2 - 1);
        const row1 = row0 + in_width;
        const row2 = row1 + in_width;

        var acc = @as(common.F32xN, @splat(bias_value));

        for (0..in_channels) |ic| {
            const input_channel = input.data[input_batch_base + ic * input_plane ..][0..input_plane];
            const weight_base = weights_channel_base + ic * 9;
            acc += loadF32xN(input_channel, row0) * @as(common.F32xN, @splat(weights.data[weight_base + 0]));
            acc += loadF32xN(input_channel, row0 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 1]));
            acc += loadF32xN(input_channel, row0 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 2]));
            acc += loadF32xN(input_channel, row1) * @as(common.F32xN, @splat(weights.data[weight_base + 3]));
            acc += loadF32xN(input_channel, row1 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 4]));
            acc += loadF32xN(input_channel, row1 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 5]));
            acc += loadF32xN(input_channel, row2) * @as(common.F32xN, @splat(weights.data[weight_base + 6]));
            acc += loadF32xN(input_channel, row2 + 1) * @as(common.F32xN, @splat(weights.data[weight_base + 7]));
            acc += loadF32xN(input_channel, row2 + 2) * @as(common.F32xN, @splat(weights.data[weight_base + 8]));
        }

        return acc;
    }

    inline fn loadF32xN(slice: []const f32, start: usize) common.F32xN {
        return .{
            slice[start + 0],
            slice[start + 2],
            slice[start + 4],
            slice[start + 6],
            slice[start + 8],
            slice[start + 10],
            slice[start + 12],
            slice[start + 14],
        };
    }

    inline fn pointPair(
        input: *const common.Tensor,
        weights: *const common.Tensor,
        input_batch_base: usize,
        weights0_channel_base: usize,
        weights1_channel_base: usize,
        oy: usize,
        ox: usize,
        bias0: f32,
        bias1: f32,
    ) [2]f32 {
        return .{
            point(input, weights, bias0, input_batch_base, weights0_channel_base, oy, ox),
            point(input, weights, bias1, input_batch_base, weights1_channel_base, oy, ox),
        };
    }

    inline fn point(
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
                    const kx0 = @as(usize, @intCast(@as(isize, @intCast(x_start)) - base_x));
                    acc += v0 * weights.data[weight_row_base + kx0] + v1 * weights.data[weight_row_base + kx0 + 1] + v2 * weights.data[weight_row_base + kx0 + 2];
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

    fn worker(task: tasks.Conv2DTask) void {
        range(task.input, task.weights, task.bias, task.output, task.options, task.oc_start, task.oc_end) catch unreachable;
    }
};
