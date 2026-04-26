const std = @import("std");
const common = @import("common.zig");
const kernel_registry = @import("../../registry/index.zig");
const grouped = @import("grouped_slice.zig");
const generic = @import("generic.zig");
const specialized_128 = @import("specialized_128.zig");


pub const q8_cache_group_size = common.q8_cache_group_size;
const handwritten_q8_head_dim = common.handwritten_q8_head_dim;
const handwritten_q8_scale_groups = common.handwritten_q8_scale_groups;
pub const dotQ8GroupedSlice = grouped.dotQ8GroupedSlice;
pub const axpyQ8GroupedSliceInPlace = grouped.axpyQ8GroupedSliceInPlace;

pub fn scaledDotProductAttentionSingleQueryQ8Cache(
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
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (key_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    if (value_cache.len != seq_len * num_key_value_heads * head_dim) return error.SizeMismatch;
    const scale_groups_per_head = std.math.divCeil(usize, head_dim, q8_cache_group_size) catch return error.InvalidDimensions;
    if (key_scales.len != seq_len * num_key_value_heads * scale_groups_per_head) return error.SizeMismatch;
    if (value_scales.len != seq_len * num_key_value_heads * scale_groups_per_head) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const entry = kernel_registry.resolve(.{ .attention_q8_decode = .{
        .head_dim = head_dim,
        .layout = .token_major,
    } });
    if (entry.shape == .qwen3_head_dim_128 and scale_groups_per_head == handwritten_q8_scale_groups) {
        return specialized_128.scaledDotProductAttentionSingleQueryQ8Cache128(
            output,
            query,
            key_cache,
            key_scales,
            value_cache,
            value_scales,
            seq_len,
            num_query_heads,
            num_key_value_heads,
            scores_scratch,
        );
    }

    return generic.scaledDotProductAttentionSingleQueryQ8CacheGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

pub fn scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
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
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    const scale_groups_per_head = std.math.divCeil(usize, head_dim, q8_cache_group_size) catch return error.InvalidDimensions;
    if (data_head_stride < seq_len * head_dim) return error.SizeMismatch;
    if (scale_head_stride < seq_len * scale_groups_per_head) return error.SizeMismatch;
    if (key_cache.len < num_key_value_heads * data_head_stride) return error.SizeMismatch;
    if (value_cache.len < num_key_value_heads * data_head_stride) return error.SizeMismatch;
    if (key_scales.len < num_key_value_heads * scale_head_stride) return error.SizeMismatch;
    if (value_scales.len < num_key_value_heads * scale_head_stride) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const entry = kernel_registry.resolve(.{ .attention_q8_decode = .{
        .head_dim = head_dim,
        .layout = .head_major,
    } });
    if (entry.shape == .qwen3_head_dim_128 and scale_groups_per_head == handwritten_q8_scale_groups) {
        return specialized_128.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor128(
            output,
            query,
            key_cache,
            key_scales,
            value_cache,
            value_scales,
            data_head_stride,
            scale_head_stride,
            seq_len,
            num_query_heads,
            num_key_value_heads,
            scores_scratch,
        );
    }

    return generic.scaledDotProductAttentionSingleQueryQ8CacheHeadMajorGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        data_head_stride,
        scale_head_stride,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

pub fn scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor(
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
    scores_scratch: []f32,
) !void {
    if (num_query_heads == 0 or num_key_value_heads == 0 or head_dim == 0) {
        return error.InvalidDimensions;
    }
    if (num_query_heads % num_key_value_heads != 0) return error.InvalidGrouping;
    if (seq_len == 0 or page_len == 0) return error.InvalidSequenceLength;
    if (output.len != num_query_heads * head_dim) return error.SizeMismatch;
    if (query.len != num_query_heads * head_dim) return error.SizeMismatch;
    const scale_groups_per_head = std.math.divCeil(usize, head_dim, q8_cache_group_size) catch return error.InvalidDimensions;
    if (page_data_stride < page_len * head_dim) return error.SizeMismatch;
    if (page_scale_stride < page_len * scale_groups_per_head) return error.SizeMismatch;
    if (head_data_stride < pages_per_head * page_data_stride) return error.SizeMismatch;
    if (head_scale_stride < pages_per_head * page_scale_stride) return error.SizeMismatch;
    if (pages_per_head * page_len < seq_len) return error.SizeMismatch;
    if (key_cache.len < num_key_value_heads * head_data_stride) return error.SizeMismatch;
    if (value_cache.len < num_key_value_heads * head_data_stride) return error.SizeMismatch;
    if (key_scales.len < num_key_value_heads * head_scale_stride) return error.SizeMismatch;
    if (value_scales.len < num_key_value_heads * head_scale_stride) return error.SizeMismatch;
    if (scores_scratch.len < seq_len) return error.InsufficientScratchSpace;

    const entry = kernel_registry.resolve(.{ .attention_q8_decode = .{
        .head_dim = head_dim,
        .layout = .paged_head_major,
    } });
    if (entry.shape == .qwen3_head_dim_128 and scale_groups_per_head == handwritten_q8_scale_groups) {
        return specialized_128.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor128(
            output,
            query,
            key_cache,
            key_scales,
            value_cache,
            value_scales,
            head_data_stride,
            head_scale_stride,
            page_data_stride,
            page_scale_stride,
            page_len,
            pages_per_head,
            seq_len,
            num_query_heads,
            num_key_value_heads,
            scores_scratch,
        );
    }

    return generic.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajorGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        head_data_stride,
        head_scale_stride,
        page_data_stride,
        page_scale_stride,
        page_len,
        pages_per_head,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

pub fn testingScaledDotProductAttentionSingleQueryQ8CacheGeneric(
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
    return generic.scaledDotProductAttentionSingleQueryQ8CacheGeneric(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        scale_groups_per_head,
        scores_scratch,
    );
}

pub fn testingScaledDotProductAttentionSingleQueryQ8Cache128(
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
    return specialized_128.scaledDotProductAttentionSingleQueryQ8Cache128(
        output,
        query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        scores_scratch,
    );
}

test {
    _ = @import("tests.zig");
}
