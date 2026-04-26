const std = @import("std");
const basic = @import("../basic.zig");
const common = @import("common.zig");
const grouped = @import("grouped_slice.zig");

const handwritten_q8_head_dim = common.handwritten_q8_head_dim;
const handwritten_q8_scale_groups = common.handwritten_q8_scale_groups;
const paired_scores_max_seq_len = common.paired_scores_max_seq_len;

const dotQ8GroupedSlice128Exact = grouped.dotQ8GroupedSlice128Exact;
const dotQ8GroupedSlice128PairExact = grouped.dotQ8GroupedSlice128PairExact;
const accumulateQ8ValueHead128 = grouped.accumulateQ8ValueHead128;
const accumulateQ8ValueHead128HeadMajor = grouped.accumulateQ8ValueHead128HeadMajor;
const accumulateQ8ValueHead128HeadMajorPair = grouped.accumulateQ8ValueHead128HeadMajorPair;
const accumulateQ8ValueHead128PagedHeadMajor = grouped.accumulateQ8ValueHead128PagedHeadMajor;
const accumulateQ8ValueHead128PagedHeadMajorPair = grouped.accumulateQ8ValueHead128PagedHeadMajorPair;

pub fn scaledDotProductAttentionSingleQueryQ8Cache128(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    scores_scratch: []f32,
) !void {
    const head_dim = handwritten_q8_head_dim;
    const scale_groups_per_head = handwritten_q8_scale_groups;
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_head_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = cache_head_index * head_dim;
            const scale_start = cache_head_index * scale_groups_per_head;
            scores[pos] = dotQ8GroupedSlice128Exact(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try basic.softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        accumulateQ8ValueHead128(
            out_slice,
            scores,
            value_cache,
            value_scales,
            num_key_value_heads,
            kv_head_idx,
        );
    }
}

pub fn scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor128(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    head_data_stride: usize,
    head_scale_stride: usize,
    page_data_stride: usize,
    page_scale_stride: usize,
    page_len: usize,
    pages_per_head: usize,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    scores_scratch: []f32,
) !void {
    const head_dim = handwritten_q8_head_dim;
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    if (group_size == 2 and seq_len <= paired_scores_max_seq_len) {
        var paired_scores: [paired_scores_max_seq_len]f32 = undefined;
        var q_head_idx: usize = 0;
        while (q_head_idx < num_query_heads) : (q_head_idx += 2) {
            const q0_start = q_head_idx * head_dim;
            const q1_start = q0_start + head_dim;
            const q0_slice = query[q0_start .. q0_start + head_dim];
            const q1_slice = query[q1_start .. q1_start + head_dim];
            const kv_head_idx = q_head_idx / group_size;
            const head_data_start = kv_head_idx * head_data_stride;
            const head_scale_start = kv_head_idx * head_scale_stride;
            const scores0 = scores_scratch[0..seq_len];
            const scores1 = paired_scores[0..seq_len];

            var pos_base: usize = 0;
            for (0..pages_per_head) |page_idx| {
                if (pos_base >= seq_len) break;
                const page_data_start = head_data_start + page_idx * page_data_stride;
                const page_scale_start = head_scale_start + page_idx * page_scale_stride;
                const page_seq_len = @min(page_len, seq_len - pos_base);
                for (0..page_seq_len) |page_offset| {
                    const cache_start = page_data_start + page_offset * head_dim;
                    const scale_start = page_scale_start + page_offset * handwritten_q8_scale_groups;
                    const pair = dotQ8GroupedSlice128PairExact(
                        q0_slice,
                        q1_slice,
                        key_cache[cache_start .. cache_start + head_dim],
                        key_scales[scale_start .. scale_start + handwritten_q8_scale_groups],
                    );
                    scores0[pos_base + page_offset] = pair[0] * scale;
                    scores1[pos_base + page_offset] = pair[1] * scale;
                }
                pos_base += page_seq_len;
            }

            try basic.softmaxInPlace(scores0);
            try basic.softmaxInPlace(scores1);

            accumulateQ8ValueHead128PagedHeadMajorPair(
                output[q0_start .. q0_start + head_dim],
                output[q1_start .. q1_start + head_dim],
                scores0,
                scores1,
                value_cache[head_data_start .. head_data_start + pages_per_head * page_data_stride],
                value_scales[head_scale_start .. head_scale_start + pages_per_head * page_scale_stride],
                page_data_stride,
                page_scale_stride,
                page_len,
            );
        }
        return;
    }

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const head_data_start = kv_head_idx * head_data_stride;
        const head_scale_start = kv_head_idx * head_scale_stride;
        const scores = scores_scratch[0..seq_len];

        var pos_base: usize = 0;
        for (0..pages_per_head) |page_idx| {
            if (pos_base >= seq_len) break;
            const page_data_start = head_data_start + page_idx * page_data_stride;
            const page_scale_start = head_scale_start + page_idx * page_scale_stride;
            const page_seq_len = @min(page_len, seq_len - pos_base);
            for (0..page_seq_len) |page_offset| {
                const cache_start = page_data_start + page_offset * head_dim;
                const scale_start = page_scale_start + page_offset * handwritten_q8_scale_groups;
                scores[pos_base + page_offset] = dotQ8GroupedSlice128Exact(
                    q_slice,
                    key_cache[cache_start .. cache_start + head_dim],
                    key_scales[scale_start .. scale_start + handwritten_q8_scale_groups],
                ) * scale;
            }
            pos_base += page_seq_len;
        }

        try basic.softmaxInPlace(scores);

        accumulateQ8ValueHead128PagedHeadMajor(
            output[q_start .. q_start + head_dim],
            scores,
            value_cache[head_data_start .. head_data_start + pages_per_head * page_data_stride],
            value_scales[head_scale_start .. head_scale_start + pages_per_head * page_scale_stride],
            page_data_stride,
            page_scale_stride,
            page_len,
        );
    }
}

pub fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajor128(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    scores_scratch: []f32,
) !void {
    const head_dim = handwritten_q8_head_dim;
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    if (group_size == 2 and seq_len <= paired_scores_max_seq_len) {
        var paired_scores: [paired_scores_max_seq_len]f32 = undefined;
        var q_head_idx: usize = 0;
        while (q_head_idx < num_query_heads) : (q_head_idx += 2) {
            const q0_start = q_head_idx * head_dim;
            const q1_start = q0_start + head_dim;
            const q0_slice = query[q0_start .. q0_start + head_dim];
            const q1_slice = query[q1_start .. q1_start + head_dim];
            const kv_head_idx = q_head_idx / group_size;
            const head_data_start = kv_head_idx * data_head_stride;
            const head_scale_start = kv_head_idx * scale_head_stride;
            const scores0 = scores_scratch[0..seq_len];
            const scores1 = paired_scores[0..seq_len];

            for (0..seq_len) |pos| {
                const cache_start = head_data_start + pos * head_dim;
                const scale_start = head_scale_start + pos * handwritten_q8_scale_groups;
                const pair = dotQ8GroupedSlice128PairExact(
                    q0_slice,
                    q1_slice,
                    key_cache[cache_start .. cache_start + head_dim],
                    key_scales[scale_start .. scale_start + handwritten_q8_scale_groups],
                );
                scores0[pos] = pair[0] * scale;
                scores1[pos] = pair[1] * scale;
            }

            try basic.softmaxInPlace(scores0);
            try basic.softmaxInPlace(scores1);

            accumulateQ8ValueHead128HeadMajorPair(
                output[q0_start .. q0_start + head_dim],
                output[q1_start .. q1_start + head_dim],
                scores0,
                scores1,
                value_cache[head_data_start .. head_data_start + seq_len * head_dim],
                value_scales[head_scale_start .. head_scale_start + seq_len * handwritten_q8_scale_groups],
            );
        }
        return;
    }

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const head_data_start = kv_head_idx * data_head_stride;
        const head_scale_start = kv_head_idx * scale_head_stride;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * handwritten_q8_scale_groups;
            scores[pos] = dotQ8GroupedSlice128Exact(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + handwritten_q8_scale_groups],
            ) * scale;
        }

        try basic.softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        accumulateQ8ValueHead128HeadMajor(
            out_slice,
            scores,
            value_cache[head_data_start .. head_data_start + seq_len * head_dim],
            value_scales[head_scale_start .. head_scale_start + seq_len * handwritten_q8_scale_groups],
        );
    }
}


