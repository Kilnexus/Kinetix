const std = @import("std");
const bfloat16 = @import("../../common/bfloat16.zig");
const common = @import("common.zig");

const q8_cache_group_size = common.q8_cache_group_size;
const handwritten_q8_head_dim = common.handwritten_q8_head_dim;
const handwritten_q8_scale_groups = common.handwritten_q8_scale_groups;

pub fn dotQ8GroupedSlice(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    std.debug.assert(lhs.len == rhs_q8.len);
    if (lhs.len == scales.len * q8_cache_group_size) {
        if (lhs.len == handwritten_q8_head_dim and scales.len == handwritten_q8_scale_groups) {
            return dotQ8GroupedSlice128Exact(lhs, rhs_q8, scales);
        }
        return dotQ8GroupedSliceExact(lhs, rhs_q8, scales);
    }

    var sum: f32 = 0.0;
    var index: usize = 0;
    for (scales) |scale_bits| {
        if (index >= lhs.len) break;
        const end = @min(lhs.len, index + q8_cache_group_size);
        const scale = bfloat16.toF32(scale_bits);
        if (end - index == 16) {
            const lhs_vec: @Vector(16, f32) = lhs[index..][0..16].*;
            const rhs_i8: @Vector(16, i8) = rhs_q8[index..][0..16].*;
            const rhs_vec: @Vector(16, f32) = @floatFromInt(rhs_i8);
            sum += @reduce(.Add, lhs_vec * rhs_vec) * scale;
        } else {
            var local: f32 = 0.0;
            var local_index = index;
            while (local_index < end) : (local_index += 1) {
                local += lhs[local_index] * @as(f32, @floatFromInt(rhs_q8[local_index]));
            }
            sum += local * scale;
        }
        index = end;
    }
    return sum;
}

pub fn axpyQ8GroupedSliceInPlace(output: []f32, alpha: f32, input_q8: []const i8, scales: []const u16) void {
    std.debug.assert(output.len == input_q8.len);
    if (output.len == scales.len * q8_cache_group_size) {
        axpyQ8GroupedSliceExactInPlace(output, alpha, input_q8, scales);
        return;
    }

    var index: usize = 0;
    for (scales) |scale_bits| {
        if (index >= output.len) break;
        const end = @min(output.len, index + q8_cache_group_size);
        const scaled_alpha = alpha * bfloat16.toF32(scale_bits);
        if (end - index == 16) {
            const alpha_vec: @Vector(16, f32) = @splat(scaled_alpha);
            const out_vec: @Vector(16, f32) = output[index..][0..16].*;
            const in_i8: @Vector(16, i8) = input_q8[index..][0..16].*;
            const in_vec: @Vector(16, f32) = @floatFromInt(in_i8);
            output[index..][0..16].* = out_vec + alpha_vec * in_vec;
        } else {
            var local_index = index;
            while (local_index < end) : (local_index += 1) {
                output[local_index] += scaled_alpha * @as(f32, @floatFromInt(input_q8[local_index]));
            }
        }
        index = end;
    }
}

fn dotQ8GroupedSliceExact(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    var sum: f32 = 0.0;
    var index: usize = 0;
    for (scales) |scale_bits| {
        sum += dotQ8Block16(lhs[index..][0..16], rhs_q8[index..][0..16], scale_bits);
        index += 16;
    }
    return sum;
}

pub fn dotQ8GroupedSlice128Exact(lhs: []const f32, rhs_q8: []const i8, scales: []const u16) f32 {
    std.debug.assert(lhs.len == handwritten_q8_head_dim);
    std.debug.assert(rhs_q8.len == handwritten_q8_head_dim);
    std.debug.assert(scales.len == handwritten_q8_scale_groups);

    return dotQ8Block16(lhs[0..16], rhs_q8[0..16], scales[0]) +
        dotQ8Block16(lhs[16..32], rhs_q8[16..32], scales[1]) +
        dotQ8Block16(lhs[32..48], rhs_q8[32..48], scales[2]) +
        dotQ8Block16(lhs[48..64], rhs_q8[48..64], scales[3]) +
        dotQ8Block16(lhs[64..80], rhs_q8[64..80], scales[4]) +
        dotQ8Block16(lhs[80..96], rhs_q8[80..96], scales[5]) +
        dotQ8Block16(lhs[96..112], rhs_q8[96..112], scales[6]) +
        dotQ8Block16(lhs[112..128], rhs_q8[112..128], scales[7]);
}

pub fn dotQ8GroupedSlice128PairExact(
    lhs0: []const f32,
    lhs1: []const f32,
    rhs_q8: []const i8,
    scales: []const u16,
) [2]f32 {
    std.debug.assert(lhs0.len == handwritten_q8_head_dim);
    std.debug.assert(lhs1.len == handwritten_q8_head_dim);
    std.debug.assert(rhs_q8.len == handwritten_q8_head_dim);
    std.debug.assert(scales.len == handwritten_q8_scale_groups);

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    inline for (0..handwritten_q8_scale_groups) |group_idx| {
        const block_start = group_idx * 16;
        const rhs_vec = loadQ8Vector16Unscaled(rhs_q8, block_start);
        const scale = bfloat16.toF32(scales[group_idx]);
        const lhs0_vec: @Vector(16, f32) = lhs0[block_start..][0..16].*;
        const lhs1_vec: @Vector(16, f32) = lhs1[block_start..][0..16].*;
        sum0 += @reduce(.Add, lhs0_vec * rhs_vec) * scale;
        sum1 += @reduce(.Add, lhs1_vec * rhs_vec) * scale;
    }
    return .{ sum0, sum1 };
}

fn axpyQ8GroupedSliceExactInPlace(output: []f32, alpha: f32, input_q8: []const i8, scales: []const u16) void {
    var index: usize = 0;
    for (scales) |scale_bits| {
        const alpha_vec: @Vector(16, f32) = @splat(alpha * bfloat16.toF32(scale_bits));
        const out_vec: @Vector(16, f32) = output[index..][0..16].*;
        const in_i8: @Vector(16, i8) = input_q8[index..][0..16].*;
        const in_vec: @Vector(16, f32) = @floatFromInt(in_i8);
        output[index..][0..16].* = out_vec + alpha_vec * in_vec;
        index += 16;
    }
}

pub fn accumulateQ8ValueHead128(
    output: []f32,
    scores: []const f32,
    value_cache: []const i8,
    value_scales: []const u16,
    num_key_value_heads: usize,
    kv_head_idx: usize,
) void {
    std.debug.assert(output.len == handwritten_q8_head_dim);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    for (scores, 0..) |weight, pos| {
        const cache_head_index = pos * num_key_value_heads + kv_head_idx;
        const cache_start = cache_head_index * handwritten_q8_head_dim;
        const scale_start = cache_head_index * handwritten_q8_scale_groups;

        acc0 += scaledQ8Vector16(value_cache, cache_start + 0, weight * bfloat16.toF32(value_scales[scale_start + 0]));
        acc1 += scaledQ8Vector16(value_cache, cache_start + 16, weight * bfloat16.toF32(value_scales[scale_start + 1]));
        acc2 += scaledQ8Vector16(value_cache, cache_start + 32, weight * bfloat16.toF32(value_scales[scale_start + 2]));
        acc3 += scaledQ8Vector16(value_cache, cache_start + 48, weight * bfloat16.toF32(value_scales[scale_start + 3]));
        acc4 += scaledQ8Vector16(value_cache, cache_start + 64, weight * bfloat16.toF32(value_scales[scale_start + 4]));
        acc5 += scaledQ8Vector16(value_cache, cache_start + 80, weight * bfloat16.toF32(value_scales[scale_start + 5]));
        acc6 += scaledQ8Vector16(value_cache, cache_start + 96, weight * bfloat16.toF32(value_scales[scale_start + 6]));
        acc7 += scaledQ8Vector16(value_cache, cache_start + 112, weight * bfloat16.toF32(value_scales[scale_start + 7]));
    }

    output[0..16].* = acc0;
    output[16..32].* = acc1;
    output[32..48].* = acc2;
    output[48..64].* = acc3;
    output[64..80].* = acc4;
    output[80..96].* = acc5;
    output[96..112].* = acc6;
    output[112..128].* = acc7;
}

pub fn accumulateQ8ValueHead128HeadMajor(
    output: []f32,
    scores: []const f32,
    value_cache_head: []const i8,
    value_scales_head: []const u16,
) void {
    std.debug.assert(output.len == handwritten_q8_head_dim);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    for (scores, 0..) |weight, pos| {
        const cache_start = pos * handwritten_q8_head_dim;
        const scale_start = pos * handwritten_q8_scale_groups;
        acc0 += scaledQ8Vector16(value_cache_head, cache_start + 0, weight * bfloat16.toF32(value_scales_head[scale_start + 0]));
        acc1 += scaledQ8Vector16(value_cache_head, cache_start + 16, weight * bfloat16.toF32(value_scales_head[scale_start + 1]));
        acc2 += scaledQ8Vector16(value_cache_head, cache_start + 32, weight * bfloat16.toF32(value_scales_head[scale_start + 2]));
        acc3 += scaledQ8Vector16(value_cache_head, cache_start + 48, weight * bfloat16.toF32(value_scales_head[scale_start + 3]));
        acc4 += scaledQ8Vector16(value_cache_head, cache_start + 64, weight * bfloat16.toF32(value_scales_head[scale_start + 4]));
        acc5 += scaledQ8Vector16(value_cache_head, cache_start + 80, weight * bfloat16.toF32(value_scales_head[scale_start + 5]));
        acc6 += scaledQ8Vector16(value_cache_head, cache_start + 96, weight * bfloat16.toF32(value_scales_head[scale_start + 6]));
        acc7 += scaledQ8Vector16(value_cache_head, cache_start + 112, weight * bfloat16.toF32(value_scales_head[scale_start + 7]));
    }

    output[0..16].* = acc0;
    output[16..32].* = acc1;
    output[32..48].* = acc2;
    output[48..64].* = acc3;
    output[64..80].* = acc4;
    output[80..96].* = acc5;
    output[96..112].* = acc6;
    output[112..128].* = acc7;
}

pub fn accumulateQ8ValueHead128HeadMajorPair(
    output0: []f32,
    output1: []f32,
    scores0: []const f32,
    scores1: []const f32,
    value_cache_head: []const i8,
    value_scales_head: []const u16,
) void {
    std.debug.assert(output0.len == handwritten_q8_head_dim);
    std.debug.assert(output1.len == handwritten_q8_head_dim);

    var acc00: @Vector(16, f32) = @splat(0.0);
    var acc01: @Vector(16, f32) = @splat(0.0);
    var acc02: @Vector(16, f32) = @splat(0.0);
    var acc03: @Vector(16, f32) = @splat(0.0);
    var acc04: @Vector(16, f32) = @splat(0.0);
    var acc05: @Vector(16, f32) = @splat(0.0);
    var acc06: @Vector(16, f32) = @splat(0.0);
    var acc07: @Vector(16, f32) = @splat(0.0);

    var acc10: @Vector(16, f32) = @splat(0.0);
    var acc11: @Vector(16, f32) = @splat(0.0);
    var acc12: @Vector(16, f32) = @splat(0.0);
    var acc13: @Vector(16, f32) = @splat(0.0);
    var acc14: @Vector(16, f32) = @splat(0.0);
    var acc15: @Vector(16, f32) = @splat(0.0);
    var acc16: @Vector(16, f32) = @splat(0.0);
    var acc17: @Vector(16, f32) = @splat(0.0);

    for (scores0, scores1, 0..) |weight0, weight1, pos| {
        const cache_start = pos * handwritten_q8_head_dim;
        const scale_start = pos * handwritten_q8_scale_groups;

        inline for (0..handwritten_q8_scale_groups) |group_idx| {
            const block_start = cache_start + group_idx * 16;
            const value_vec = loadQ8Vector16Unscaled(value_cache_head, block_start);
            const scale = bfloat16.toF32(value_scales_head[scale_start + group_idx]);
            const scaled0: @Vector(16, f32) = @splat(weight0 * scale);
            const scaled1: @Vector(16, f32) = @splat(weight1 * scale);
            switch (group_idx) {
                0 => {
                    acc00 += value_vec * scaled0;
                    acc10 += value_vec * scaled1;
                },
                1 => {
                    acc01 += value_vec * scaled0;
                    acc11 += value_vec * scaled1;
                },
                2 => {
                    acc02 += value_vec * scaled0;
                    acc12 += value_vec * scaled1;
                },
                3 => {
                    acc03 += value_vec * scaled0;
                    acc13 += value_vec * scaled1;
                },
                4 => {
                    acc04 += value_vec * scaled0;
                    acc14 += value_vec * scaled1;
                },
                5 => {
                    acc05 += value_vec * scaled0;
                    acc15 += value_vec * scaled1;
                },
                6 => {
                    acc06 += value_vec * scaled0;
                    acc16 += value_vec * scaled1;
                },
                7 => {
                    acc07 += value_vec * scaled0;
                    acc17 += value_vec * scaled1;
                },
                else => unreachable,
            }
        }
    }

    output0[0..16].* = acc00;
    output0[16..32].* = acc01;
    output0[32..48].* = acc02;
    output0[48..64].* = acc03;
    output0[64..80].* = acc04;
    output0[80..96].* = acc05;
    output0[96..112].* = acc06;
    output0[112..128].* = acc07;

    output1[0..16].* = acc10;
    output1[16..32].* = acc11;
    output1[32..48].* = acc12;
    output1[48..64].* = acc13;
    output1[64..80].* = acc14;
    output1[80..96].* = acc15;
    output1[96..112].* = acc16;
    output1[112..128].* = acc17;
}

pub fn accumulateQ8ValueHead128PagedHeadMajor(
    output: []f32,
    scores: []const f32,
    value_cache_head: []const i8,
    value_scales_head: []const u16,
    page_data_stride: usize,
    page_scale_stride: usize,
    page_len: usize,
) void {
    std.debug.assert(output.len == handwritten_q8_head_dim);

    var acc0: @Vector(16, f32) = @splat(0.0);
    var acc1: @Vector(16, f32) = @splat(0.0);
    var acc2: @Vector(16, f32) = @splat(0.0);
    var acc3: @Vector(16, f32) = @splat(0.0);
    var acc4: @Vector(16, f32) = @splat(0.0);
    var acc5: @Vector(16, f32) = @splat(0.0);
    var acc6: @Vector(16, f32) = @splat(0.0);
    var acc7: @Vector(16, f32) = @splat(0.0);

    var pos_base: usize = 0;
    var page_idx: usize = 0;
    while (pos_base < scores.len) : (page_idx += 1) {
        const page_data_start = page_idx * page_data_stride;
        const page_scale_start = page_idx * page_scale_stride;
        const page_seq_len = @min(page_len, scores.len - pos_base);
        for (0..page_seq_len) |page_offset| {
            const weight = scores[pos_base + page_offset];
            const cache_start = page_data_start + page_offset * handwritten_q8_head_dim;
            const scale_start = page_scale_start + page_offset * handwritten_q8_scale_groups;
            acc0 += scaledQ8Vector16(value_cache_head, cache_start + 0, weight * bfloat16.toF32(value_scales_head[scale_start + 0]));
            acc1 += scaledQ8Vector16(value_cache_head, cache_start + 16, weight * bfloat16.toF32(value_scales_head[scale_start + 1]));
            acc2 += scaledQ8Vector16(value_cache_head, cache_start + 32, weight * bfloat16.toF32(value_scales_head[scale_start + 2]));
            acc3 += scaledQ8Vector16(value_cache_head, cache_start + 48, weight * bfloat16.toF32(value_scales_head[scale_start + 3]));
            acc4 += scaledQ8Vector16(value_cache_head, cache_start + 64, weight * bfloat16.toF32(value_scales_head[scale_start + 4]));
            acc5 += scaledQ8Vector16(value_cache_head, cache_start + 80, weight * bfloat16.toF32(value_scales_head[scale_start + 5]));
            acc6 += scaledQ8Vector16(value_cache_head, cache_start + 96, weight * bfloat16.toF32(value_scales_head[scale_start + 6]));
            acc7 += scaledQ8Vector16(value_cache_head, cache_start + 112, weight * bfloat16.toF32(value_scales_head[scale_start + 7]));
        }
        pos_base += page_seq_len;
    }

    output[0..16].* = acc0;
    output[16..32].* = acc1;
    output[32..48].* = acc2;
    output[48..64].* = acc3;
    output[64..80].* = acc4;
    output[80..96].* = acc5;
    output[96..112].* = acc6;
    output[112..128].* = acc7;
}

pub fn accumulateQ8ValueHead128PagedHeadMajorPair(
    output0: []f32,
    output1: []f32,
    scores0: []const f32,
    scores1: []const f32,
    value_cache_head: []const i8,
    value_scales_head: []const u16,
    page_data_stride: usize,
    page_scale_stride: usize,
    page_len: usize,
) void {
    std.debug.assert(output0.len == handwritten_q8_head_dim);
    std.debug.assert(output1.len == handwritten_q8_head_dim);

    var acc00: @Vector(16, f32) = @splat(0.0);
    var acc01: @Vector(16, f32) = @splat(0.0);
    var acc02: @Vector(16, f32) = @splat(0.0);
    var acc03: @Vector(16, f32) = @splat(0.0);
    var acc04: @Vector(16, f32) = @splat(0.0);
    var acc05: @Vector(16, f32) = @splat(0.0);
    var acc06: @Vector(16, f32) = @splat(0.0);
    var acc07: @Vector(16, f32) = @splat(0.0);
    var acc10: @Vector(16, f32) = @splat(0.0);
    var acc11: @Vector(16, f32) = @splat(0.0);
    var acc12: @Vector(16, f32) = @splat(0.0);
    var acc13: @Vector(16, f32) = @splat(0.0);
    var acc14: @Vector(16, f32) = @splat(0.0);
    var acc15: @Vector(16, f32) = @splat(0.0);
    var acc16: @Vector(16, f32) = @splat(0.0);
    var acc17: @Vector(16, f32) = @splat(0.0);

    var pos_base: usize = 0;
    var page_idx: usize = 0;
    while (pos_base < scores0.len) : (page_idx += 1) {
        const page_data_start = page_idx * page_data_stride;
        const page_scale_start = page_idx * page_scale_stride;
        const page_seq_len = @min(page_len, scores0.len - pos_base);
        for (0..page_seq_len) |page_offset| {
            const weight0 = scores0[pos_base + page_offset];
            const weight1 = scores1[pos_base + page_offset];
            const cache_start = page_data_start + page_offset * handwritten_q8_head_dim;
            const scale_start = page_scale_start + page_offset * handwritten_q8_scale_groups;

            inline for (0..handwritten_q8_scale_groups) |group_idx| {
                const block_start = cache_start + group_idx * 16;
                const value_vec = loadQ8Vector16Unscaled(value_cache_head, block_start);
                const scale = bfloat16.toF32(value_scales_head[scale_start + group_idx]);
                const scaled0: @Vector(16, f32) = @splat(weight0 * scale);
                const scaled1: @Vector(16, f32) = @splat(weight1 * scale);
                switch (group_idx) {
                    0 => {
                        acc00 += value_vec * scaled0;
                        acc10 += value_vec * scaled1;
                    },
                    1 => {
                        acc01 += value_vec * scaled0;
                        acc11 += value_vec * scaled1;
                    },
                    2 => {
                        acc02 += value_vec * scaled0;
                        acc12 += value_vec * scaled1;
                    },
                    3 => {
                        acc03 += value_vec * scaled0;
                        acc13 += value_vec * scaled1;
                    },
                    4 => {
                        acc04 += value_vec * scaled0;
                        acc14 += value_vec * scaled1;
                    },
                    5 => {
                        acc05 += value_vec * scaled0;
                        acc15 += value_vec * scaled1;
                    },
                    6 => {
                        acc06 += value_vec * scaled0;
                        acc16 += value_vec * scaled1;
                    },
                    7 => {
                        acc07 += value_vec * scaled0;
                        acc17 += value_vec * scaled1;
                    },
                    else => unreachable,
                }
            }
        }
        pos_base += page_seq_len;
    }

    output0[0..16].* = acc00;
    output0[16..32].* = acc01;
    output0[32..48].* = acc02;
    output0[48..64].* = acc03;
    output0[64..80].* = acc04;
    output0[80..96].* = acc05;
    output0[96..112].* = acc06;
    output0[112..128].* = acc07;

    output1[0..16].* = acc10;
    output1[16..32].* = acc11;
    output1[32..48].* = acc12;
    output1[48..64].* = acc13;
    output1[64..80].* = acc14;
    output1[80..96].* = acc15;
    output1[96..112].* = acc16;
    output1[112..128].* = acc17;
}

fn scaledQ8Vector16(input_q8: []const i8, start: usize, scale: f32) @Vector(16, f32) {
    const scale_vec: @Vector(16, f32) = @splat(scale);
    return loadQ8Vector16Unscaled(input_q8, start) * scale_vec;
}

fn loadQ8Vector16Unscaled(input_q8: []const i8, start: usize) @Vector(16, f32) {
    const input_i8: @Vector(16, i8) = input_q8[start..][0..16].*;
    return @floatFromInt(input_i8);
}

fn dotQ8Block16(lhs: []const f32, rhs_q8: []const i8, scale_bits: u16) f32 {
    const lhs_vec: @Vector(16, f32) = lhs[0..16].*;
    const rhs_i8: @Vector(16, i8) = rhs_q8[0..16].*;
    const rhs_vec: @Vector(16, f32) = @floatFromInt(rhs_i8);
    return @reduce(.Add, lhs_vec * rhs_vec) * bfloat16.toF32(scale_bits);
}
