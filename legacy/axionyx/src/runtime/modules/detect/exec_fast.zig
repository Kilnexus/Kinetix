const std = @import("std");
const detect_types = @import("types.zig");
const thread_pool = @import("../../thread_pool.zig");

const Tensor = detect_types.Tensor;
const ConvPlan = detect_types.ConvPlan;
const max_detect_fast_threads = detect_types.max_detect_fast_threads;

pub fn canUseFastDetectCv2Conv(input: *const Tensor, plan: *const ConvPlan) bool {
    return input.shape[0] == 1 and
        plan.weight.shape[0] == 64 and
        plan.weight.shape[2] == 3 and
        plan.weight.shape[3] == 3 and
        plan.stride_h == 1 and
        plan.stride_w == 1 and
        plan.pad_h == 1 and
        plan.pad_w == 1 and
        plan.groups == 1 and
        plan.activation == .silu;
}

pub fn canUseFastDetectCv3DepthwiseConv(input: *const Tensor, plan: *const ConvPlan) bool {
    return input.shape[0] == 1 and
        plan.weight.shape[2] == 3 and
        plan.weight.shape[3] == 3 and
        plan.stride_h == 1 and
        plan.stride_w == 1 and
        plan.pad_h == 1 and
        plan.pad_w == 1 and
        plan.activation == .silu and
        plan.groups == input.shape[1] and
        plan.weight.shape[1] == 1 and
        plan.weight.shape[0] == input.shape[1];
}

pub fn runDetectFast3x3Conv64Batch1(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
) !Tensor {
    const thread_count = chooseDetectFastConvThreadCount(input.shape[2] * input.shape[3]);
    if (thread_count > 1) {
        return runDetectFast3x3Conv64Batch1Parallel(allocator, input, plan, thread_count);
    }
    return runDetectFast3x3Conv64Batch1Range(allocator, input, plan, 0, 64);
}

pub fn runDetectFastDepthwise3x3Batch1(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
) !Tensor {
    const channels = input.shape[1];
    const thread_count = @min(chooseDetectFastDepthwiseThreadCount(channels, input.shape[2] * input.shape[3]), channels);
    if (thread_count > 1) {
        return runDetectFastDepthwise3x3Batch1Parallel(allocator, input, plan, thread_count);
    }
    return runDetectFastDepthwise3x3Batch1Range(allocator, input, plan, 0, channels);
}

const DetectFastConvTask = struct {
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    oc_start: usize,
    oc_end: usize,
};

const DetectFastDepthwiseTask = struct {
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    channel_start: usize,
    channel_end: usize,
};

const DetectPairAcc = struct {
    a: f32,
    b: f32,
};

const max_detect_fast_interior_width = 128;

fn chooseDetectFastConvThreadCount(spatial: usize) usize {
    if (spatial >= 128) return 2;
    return 1;
}

fn chooseDetectFastDepthwiseThreadCount(channels: usize, spatial: usize) usize {
    if (channels >= 32 and spatial >= 128) return 2;
    return 1;
}

fn runDetectFast3x3Conv64Batch1Parallel(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    thread_count: usize,
) !Tensor {
    var output = try Tensor.init(allocator, 1, 64, input.shape[2], input.shape[3]);
    errdefer output.deinit();

    if (thread_pool.get()) |pool| {
        var wg: std.Thread.WaitGroup = .{};
        for (0..thread_count) |thread_index| {
            const oc_start = (64 * thread_index) / thread_count;
            const oc_end = (64 * (thread_index + 1)) / thread_count;
            if (oc_start == oc_end) continue;

            const task = DetectFastConvTask{
                .input = input,
                .plan = plan,
                .output = &output,
                .oc_start = oc_start,
                .oc_end = oc_end,
            };
            if (thread_index + 1 == thread_count) {
                runDetectFast3x3Conv64Batch1Worker(task);
            } else {
                pool.spawnWg(&wg, runDetectFast3x3Conv64Batch1Worker, .{task});
            }
        }
        wg.wait();
        return output;
    }

    var threads: [max_detect_fast_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..thread_count) |thread_index| {
        const oc_start = (64 * thread_index) / thread_count;
        const oc_end = (64 * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, runDetectFast3x3Conv64Batch1Worker, .{
                DetectFastConvTask{
                    .input = input,
                    .plan = plan,
                    .output = &output,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
    return output;
}

fn runDetectFast3x3Conv64Batch1Range(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    oc_start: usize,
    oc_end: usize,
) !Tensor {
    var output = try Tensor.init(allocator, 1, 64, input.shape[2], input.shape[3]);
    errdefer output.deinit();
    runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
    return output;
}

fn runDetectFast3x3Conv64Batch1Into(
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    oc_start: usize,
    oc_end: usize,
) void {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_height = input.shape[2];
    const out_width = input.shape[3];
    const input_plane = in_height * in_width;
    const output_plane = out_height * out_width;
    const interior_h_end = if (out_height > 1) out_height - 1 else 0;
    const interior_w_end = if (out_width > 1) out_width - 1 else 0;
    const interior_count = if (interior_w_end > 1) interior_w_end - 1 else 0;
    const use_row_accumulators = interior_count > 0 and interior_count <= max_detect_fast_interior_width;
    const weight_data = plan.weight.data;
    var acc0_buf: [max_detect_fast_interior_width]f32 = undefined;
    var acc1_buf: [max_detect_fast_interior_width]f32 = undefined;

    var oc: usize = oc_start;

    while (oc + 1 < oc_end) : (oc += 2) {
        const weight0_base = oc * in_channels * 9;
        const weight1_base = (oc + 1) * in_channels * 9;
        const out0_base = oc * output_plane;
        const out1_base = (oc + 1) * output_plane;
        const bias0: f32 = if (plan.bias) |b| b[oc] else 0.0;
        const bias1: f32 = if (plan.bias) |b| b[oc + 1] else 0.0;

        for (0..out_height) |oy| {
            const out0_row = out0_base + oy * out_width;
            const out1_row = out1_base + oy * out_width;

            if (oy == 0 or oy >= interior_h_end) {
                for (0..out_width) |ox| {
                    const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, ox);
                    output.data[out0_row + ox] = silu(acc.a);
                    output.data[out1_row + ox] = silu(acc.b);
                }
                continue;
            }

            {
                const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, 0);
                output.data[out0_row] = silu(acc.a);
                output.data[out1_row] = silu(acc.b);
            }

            if (use_row_accumulators) {
                @memset(acc0_buf[0..interior_count], bias0);
                @memset(acc1_buf[0..interior_count], bias1);
                const row0_base = (oy - 1) * in_width;
                const row1_base = row0_base + in_width;
                const row2_base = row1_base + in_width;

                for (0..in_channels) |ic| {
                    const input_channel = input.data[ic * input_plane ..][0..input_plane];
                    const ic_weight0 = weight0_base + ic * 9;
                    const ic_weight1 = weight1_base + ic * 9;
                    const w0: @Vector(8, f32) = weight_data[ic_weight0..][0..8].*;
                    const w1: @Vector(8, f32) = weight_data[ic_weight1..][0..8].*;

                    var v00 = input_channel[row0_base];
                    var v01 = input_channel[row0_base + 1];
                    var v02 = input_channel[row0_base + 2];
                    var v10 = input_channel[row1_base];
                    var v11 = input_channel[row1_base + 1];
                    var v12 = input_channel[row1_base + 2];
                    var v20 = input_channel[row2_base];
                    var v21 = input_channel[row2_base + 1];
                    var v22 = input_channel[row2_base + 2];

                    for (0..interior_count) |ix| {
                        const src8: @Vector(8, f32) = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                        acc0_buf[ix] += @reduce(.Add, src8 * w0) + v22 * weight_data[ic_weight0 + 8];
                        acc1_buf[ix] += @reduce(.Add, src8 * w1) + v22 * weight_data[ic_weight1 + 8];

                        if (ix + 1 < interior_count) {
                            const next_col = ix + 3;
                            v00 = v01;
                            v01 = v02;
                            v02 = input_channel[row0_base + next_col];
                            v10 = v11;
                            v11 = v12;
                            v12 = input_channel[row1_base + next_col];
                            v20 = v21;
                            v21 = v22;
                            v22 = input_channel[row2_base + next_col];
                        }
                    }
                }

                for (0..interior_count) |ix| {
                    const ox = ix + 1;
                    output.data[out0_row + ox] = silu(acc0_buf[ix]);
                    output.data[out1_row + ox] = silu(acc1_buf[ix]);
                }
            } else {
                for (1..interior_w_end) |ox| {
                    var acc0: f32 = bias0;
                    var acc1: f32 = bias1;
                    const row0 = (oy - 1) * in_width + (ox - 1);
                    const row1 = row0 + in_width;
                    const row2 = row1 + in_width;

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[ic * input_plane ..][0..input_plane];
                        const ic_weight0 = weight0_base + ic * 9;
                        const ic_weight1 = weight1_base + ic * 9;

                        const v00 = input_channel[row0];
                        const v01 = input_channel[row0 + 1];
                        const v02 = input_channel[row0 + 2];
                        const v10 = input_channel[row1];
                        const v11 = input_channel[row1 + 1];
                        const v12 = input_channel[row1 + 2];
                        const v20 = input_channel[row2];
                        const v21 = input_channel[row2 + 1];
                        const v22 = input_channel[row2 + 2];
                        const src8: @Vector(8, f32) = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                        const w0: @Vector(8, f32) = weight_data[ic_weight0..][0..8].*;
                        const w1: @Vector(8, f32) = weight_data[ic_weight1..][0..8].*;

                        acc0 += @reduce(.Add, src8 * w0);
                        acc0 += v22 * weight_data[ic_weight0 + 8];
                        acc1 += @reduce(.Add, src8 * w1);
                        acc1 += v22 * weight_data[ic_weight1 + 8];
                    }

                    output.data[out0_row + ox] = silu(acc0);
                    output.data[out1_row + ox] = silu(acc1);
                }
            }

            for (interior_w_end..out_width) |ox| {
                const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, ox);
                output.data[out0_row + ox] = silu(acc.a);
                output.data[out1_row + ox] = silu(acc.b);
            }
        }
    }

    while (oc < oc_end) : (oc += 1) {
        const weight_base = oc * in_channels * 9;
        const out_base = oc * output_plane;
        const bias_value: f32 = if (plan.bias) |b| b[oc] else 0.0;

        for (0..out_height) |oy| {
            const out_row = out_base + oy * out_width;
            if (oy == 0 or oy >= interior_h_end) {
                for (0..out_width) |ox| {
                    output.data[out_row + ox] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, ox));
                }
                continue;
            }

            output.data[out_row] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, 0));

            if (use_row_accumulators) {
                @memset(acc0_buf[0..interior_count], bias_value);
                const row0_base = (oy - 1) * in_width;
                const row1_base = row0_base + in_width;
                const row2_base = row1_base + in_width;

                for (0..in_channels) |ic| {
                    const input_channel = input.data[ic * input_plane ..][0..input_plane];
                    const ic_weight = weight_base + ic * 9;
                    const w: @Vector(8, f32) = weight_data[ic_weight..][0..8].*;

                    var v00 = input_channel[row0_base];
                    var v01 = input_channel[row0_base + 1];
                    var v02 = input_channel[row0_base + 2];
                    var v10 = input_channel[row1_base];
                    var v11 = input_channel[row1_base + 1];
                    var v12 = input_channel[row1_base + 2];
                    var v20 = input_channel[row2_base];
                    var v21 = input_channel[row2_base + 1];
                    var v22 = input_channel[row2_base + 2];

                    for (0..interior_count) |ix| {
                        const src8: @Vector(8, f32) = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                        acc0_buf[ix] += @reduce(.Add, src8 * w) + v22 * weight_data[ic_weight + 8];

                        if (ix + 1 < interior_count) {
                            const next_col = ix + 3;
                            v00 = v01;
                            v01 = v02;
                            v02 = input_channel[row0_base + next_col];
                            v10 = v11;
                            v11 = v12;
                            v12 = input_channel[row1_base + next_col];
                            v20 = v21;
                            v21 = v22;
                            v22 = input_channel[row2_base + next_col];
                        }
                    }
                }

                for (0..interior_count) |ix| {
                    output.data[out_row + ix + 1] = silu(acc0_buf[ix]);
                }
            } else {
                for (1..interior_w_end) |ox| {
                    var acc: f32 = bias_value;
                    const row0 = (oy - 1) * in_width + (ox - 1);
                    const row1 = row0 + in_width;
                    const row2 = row1 + in_width;

                    for (0..in_channels) |ic| {
                        const input_channel = input.data[ic * input_plane ..][0..input_plane];
                        const ic_weight = weight_base + ic * 9;
                        const v00 = input_channel[row0];
                        const v01 = input_channel[row0 + 1];
                        const v02 = input_channel[row0 + 2];
                        const v10 = input_channel[row1];
                        const v11 = input_channel[row1 + 1];
                        const v12 = input_channel[row1 + 2];
                        const v20 = input_channel[row2];
                        const v21 = input_channel[row2 + 1];
                        const src8: @Vector(8, f32) = .{ v00, v01, v02, v10, v11, v12, v20, v21 };
                        const w: @Vector(8, f32) = weight_data[ic_weight..][0..8].*;
                        acc += @reduce(.Add, src8 * w);
                        acc += input_channel[row2 + 2] * weight_data[ic_weight + 8];
                    }

                    output.data[out_row + ox] = silu(acc);
                }
            }

            for (interior_w_end..out_width) |ox| {
                output.data[out_row + ox] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, ox));
            }
        }
    }
}

fn runDetectFast3x3Conv64Batch1Worker(task: DetectFastConvTask) void {
    runDetectFast3x3Conv64Batch1Into(task.input, task.plan, task.output, task.oc_start, task.oc_end);
}

fn runDetectFastDepthwise3x3Batch1Parallel(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    thread_count: usize,
) !Tensor {
    var output = try Tensor.init(allocator, 1, input.shape[1], input.shape[2], input.shape[3]);
    errdefer output.deinit();

    if (thread_pool.get()) |pool| {
        const channels = input.shape[1];
        var wg: std.Thread.WaitGroup = .{};
        for (0..thread_count) |thread_index| {
            const c_start = (channels * thread_index) / thread_count;
            const c_end = (channels * (thread_index + 1)) / thread_count;
            if (c_start == c_end) continue;

            const task = DetectFastDepthwiseTask{
                .input = input,
                .plan = plan,
                .output = &output,
                .channel_start = c_start,
                .channel_end = c_end,
            };
            if (thread_index + 1 == thread_count) {
                runDetectFastDepthwise3x3Batch1Worker(task);
            } else {
                pool.spawnWg(&wg, runDetectFastDepthwise3x3Batch1Worker, .{task});
            }
        }
        wg.wait();
        return output;
    }

    var threads: [max_detect_fast_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;
    const channels = input.shape[1];

    for (0..thread_count) |thread_index| {
        const c_start = (channels * thread_index) / thread_count;
        const c_end = (channels * (thread_index + 1)) / thread_count;
        if (c_start == c_end) continue;

        if (thread_index + 1 == thread_count) {
            runDetectFastDepthwise3x3Batch1Into(input, plan, &output, c_start, c_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, runDetectFastDepthwise3x3Batch1Worker, .{
                DetectFastDepthwiseTask{
                    .input = input,
                    .plan = plan,
                    .output = &output,
                    .channel_start = c_start,
                    .channel_end = c_end,
                },
            }) catch {
                runDetectFastDepthwise3x3Batch1Into(input, plan, &output, c_start, c_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
    return output;
}

fn runDetectFastDepthwise3x3Batch1Range(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    channel_start: usize,
    channel_end: usize,
) !Tensor {
    var output = try Tensor.init(allocator, 1, input.shape[1], input.shape[2], input.shape[3]);
    errdefer output.deinit();
    runDetectFastDepthwise3x3Batch1Into(input, plan, &output, channel_start, channel_end);
    return output;
}

fn runDetectFastDepthwise3x3Batch1Into(
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    channel_start: usize,
    channel_end: usize,
) void {
    const height = input.shape[2];
    const width = input.shape[3];
    const plane = height * width;
    const interior_h_end = if (height > 1) height - 1 else 0;
    const interior_w_end = if (width > 1) width - 1 else 0;

    for (channel_start..channel_end) |c| {
        const src = input.data[c * plane ..][0..plane];
        const dst = output.data[c * plane ..][0..plane];
        const w = plan.weight.data[c * 9 ..][0..9];
        const bias_value: f32 = if (plan.bias) |bias| bias[c] else 0.0;

        for (0..height) |y| {
            const out_row = y * width;
            if (y == 0 or y >= interior_h_end) {
                for (0..width) |x| {
                    dst[out_row + x] = silu(detectFastDepthwise3x3Point(src, width, height, w, bias_value, y, x));
                }
                continue;
            }

            dst[out_row] = silu(detectFastDepthwise3x3Point(src, width, height, w, bias_value, y, 0));

            for (1..interior_w_end) |x| {
                const row0 = (y - 1) * width + (x - 1);
                const row1 = row0 + width;
                const row2 = row1 + width;
                var acc = bias_value;
                const src8: @Vector(8, f32) = .{
                    src[row0],
                    src[row0 + 1],
                    src[row0 + 2],
                    src[row1],
                    src[row1 + 1],
                    src[row1 + 2],
                    src[row2],
                    src[row2 + 1],
                };
                const w8: @Vector(8, f32) = w[0..8].*;
                acc += @reduce(.Add, src8 * w8);
                acc += src[row2 + 2] * w[8];
                dst[out_row + x] = silu(acc);
            }

            for (interior_w_end..width) |x| {
                dst[out_row + x] = silu(detectFastDepthwise3x3Point(src, width, height, w, bias_value, y, x));
            }
        }
    }
}

fn detectFastDepthwise3x3Point(
    src: []const f32,
    width: usize,
    height: usize,
    weights: []const f32,
    bias_value: f32,
    oy: usize,
    ox: usize,
) f32 {
    const base_y = @as(isize, @intCast(oy)) - 1;
    const base_x = @as(isize, @intCast(ox)) - 1;
    const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
    const y_end: usize = @intCast(@min(@as(isize, @intCast(height)), base_y + 3));
    const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
    const x_end: usize = @intCast(@min(@as(isize, @intCast(width)), base_x + 3));

    var acc = bias_value;
    var iy = y_start;
    while (iy < y_end) : (iy += 1) {
        const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
        const row_base = iy * width;
        const weight_row = ky * 3;
        var ix = x_start;
        if (x_end - x_start == 3) {
            const kx0 = @as(usize, @intCast(@as(isize, @intCast(x_start)) - base_x));
            acc += src[row_base + x_start] * weights[weight_row + kx0];
            acc += src[row_base + x_start + 1] * weights[weight_row + kx0 + 1];
            acc += src[row_base + x_start + 2] * weights[weight_row + kx0 + 2];
            continue;
        }
        while (ix < x_end) : (ix += 1) {
            const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
            acc += src[row_base + ix] * weights[weight_row + kx];
        }
    }
    return acc;
}

fn runDetectFastDepthwise3x3Batch1Worker(task: DetectFastDepthwiseTask) void {
    runDetectFastDepthwise3x3Batch1Into(task.input, task.plan, task.output, task.channel_start, task.channel_end);
}

inline fn detectFast3x3PointPair(
    input: *const Tensor,
    weights: *const Tensor,
    bias0: f32,
    bias1: f32,
    weight0_base: usize,
    weight1_base: usize,
    oy: usize,
    ox: usize,
) DetectPairAcc {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const input_plane = in_height * in_width;
    const base_y = @as(isize, @intCast(oy)) - 1;
    const base_x = @as(isize, @intCast(ox)) - 1;

    var acc0: f32 = bias0;
    var acc1: f32 = bias1;
    for (0..in_channels) |ic| {
        const input_channel = input.data[ic * input_plane ..][0..input_plane];
        const ic_weight0 = weight0_base + ic * 9;
        const ic_weight1 = weight1_base + ic * 9;
        const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
        const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
        const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
        const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

        var iy = y_start;
        while (iy < y_end) : (iy += 1) {
            const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
            const input_row_base = iy * in_width;
            const weight0_row = ic_weight0 + ky * 3;
            const weight1_row = ic_weight1 + ky * 3;
            var ix = x_start;
            if (x_end - x_start == 3) {
                const kx0 = @as(usize, @intCast(@as(isize, @intCast(x_start)) - base_x));
                const s0 = input_channel[input_row_base + x_start];
                const s1 = input_channel[input_row_base + x_start + 1];
                const s2 = input_channel[input_row_base + x_start + 2];
                acc0 += s0 * weights.data[weight0_row + kx0] + s1 * weights.data[weight0_row + kx0 + 1] + s2 * weights.data[weight0_row + kx0 + 2];
                acc1 += s0 * weights.data[weight1_row + kx0] + s1 * weights.data[weight1_row + kx0 + 1] + s2 * weights.data[weight1_row + kx0 + 2];
                continue;
            }
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                const src = input_channel[input_row_base + ix];
                acc0 += src * weights.data[weight0_row + kx];
                acc1 += src * weights.data[weight1_row + kx];
            }
        }
    }
    return .{ .a = acc0, .b = acc1 };
}

inline fn detectFast3x3Point(
    input: *const Tensor,
    weights: *const Tensor,
    bias_value: f32,
    weight_base: usize,
    oy: usize,
    ox: usize,
) f32 {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const input_plane = in_height * in_width;
    const base_y = @as(isize, @intCast(oy)) - 1;
    const base_x = @as(isize, @intCast(ox)) - 1;

    var acc: f32 = bias_value;
    for (0..in_channels) |ic| {
        const input_channel = input.data[ic * input_plane ..][0..input_plane];
        const ic_weight = weight_base + ic * 9;
        const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
        const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
        const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
        const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

        var iy = y_start;
        while (iy < y_end) : (iy += 1) {
            const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
            const input_row_base = iy * in_width;
            const weight_row = ic_weight + ky * 3;
            var ix = x_start;
            if (x_end - x_start == 3) {
                const kx0 = @as(usize, @intCast(@as(isize, @intCast(x_start)) - base_x));
                acc += input_channel[input_row_base + x_start] * weights.data[weight_row + kx0];
                acc += input_channel[input_row_base + x_start + 1] * weights.data[weight_row + kx0 + 1];
                acc += input_channel[input_row_base + x_start + 2] * weights.data[weight_row + kx0 + 2];
                continue;
            }
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                acc += input_channel[input_row_base + ix] * weights.data[weight_row + kx];
            }
        }
    }
    return acc;
}

inline fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}
