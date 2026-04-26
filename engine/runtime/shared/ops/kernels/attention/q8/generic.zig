const std = @import("std");
const basic = @import("../basic.zig");
const grouped = @import("grouped_slice.zig");

const dotQ8GroupedSlice = grouped.dotQ8GroupedSlice;
const axpyQ8GroupedSliceInPlace = grouped.axpyQ8GroupedSliceInPlace;

pub fn scaledDotProductAttentionSingleQueryQ8CacheGeneric(
    output: []f32,
    query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    num_query_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    scale_groups_per_head: usize,
    scores_scratch: []f32,
) !void {
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
            scores[pos] = dotQ8GroupedSlice(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try basic.softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_head_index = pos * num_key_value_heads + kv_head_idx;
            const cache_start = cache_head_index * head_dim;
            const scale_start = cache_head_index * scale_groups_per_head;
            axpyQ8GroupedSliceInPlace(
                out_slice,
                weight,
                value_cache[cache_start .. cache_start + head_dim],
                value_scales[scale_start .. scale_start + scale_groups_per_head],
            );
        }
    }
}

pub fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajorGeneric(
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
    head_dim: usize,
    scale_groups_per_head: usize,
    scores_scratch: []f32,
) !void {
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

    for (0..num_query_heads) |q_head_idx| {
        const q_start = q_head_idx * head_dim;
        const q_slice = query[q_start .. q_start + head_dim];
        const kv_head_idx = q_head_idx / group_size;
        const head_data_start = kv_head_idx * data_head_stride;
        const head_scale_start = kv_head_idx * scale_head_stride;
        const scores = scores_scratch[0..seq_len];

        for (0..seq_len) |pos| {
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * scale_groups_per_head;
            scores[pos] = dotQ8GroupedSlice(
                q_slice,
                key_cache[cache_start .. cache_start + head_dim],
                key_scales[scale_start .. scale_start + scale_groups_per_head],
            ) * scale;
        }

        try basic.softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        for (0..seq_len) |pos| {
            const weight = scores[pos];
            const cache_start = head_data_start + pos * head_dim;
            const scale_start = head_scale_start + pos * scale_groups_per_head;
            axpyQ8GroupedSliceInPlace(
                out_slice,
                weight,
                value_cache[cache_start .. cache_start + head_dim],
                value_scales[scale_start .. scale_start + scale_groups_per_head],
            );
        }
    }
}

pub fn scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajorGeneric(
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
    head_dim: usize,
    scale_groups_per_head: usize,
    scores_scratch: []f32,
) !void {
    const group_size = num_query_heads / num_key_value_heads;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    @memset(output, 0.0);

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
                const scale_start = page_scale_start + page_offset * scale_groups_per_head;
                scores[pos_base + page_offset] = dotQ8GroupedSlice(
                    q_slice,
                    key_cache[cache_start .. cache_start + head_dim],
                    key_scales[scale_start .. scale_start + scale_groups_per_head],
                ) * scale;
            }
            pos_base += page_seq_len;
        }

        try basic.softmaxInPlace(scores);

        const out_slice = output[q_start .. q_start + head_dim];
        pos_base = 0;
        for (0..pages_per_head) |page_idx| {
            if (pos_base >= seq_len) break;
            const page_data_start = head_data_start + page_idx * page_data_stride;
            const page_scale_start = head_scale_start + page_idx * page_scale_stride;
            const page_seq_len = @min(page_len, seq_len - pos_base);
            for (0..page_seq_len) |page_offset| {
                const cache_start = page_data_start + page_offset * head_dim;
                const scale_start = page_scale_start + page_offset * scale_groups_per_head;
                axpyQ8GroupedSliceInPlace(
                    out_slice,
                    scores[pos_base + page_offset],
                    value_cache[cache_start .. cache_start + head_dim],
                    value_scales[scale_start .. scale_start + scale_groups_per_head],
                );
            }
            pos_base += page_seq_len;
        }
    }
}

