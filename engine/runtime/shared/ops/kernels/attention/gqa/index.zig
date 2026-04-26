const basic = @import("../basic.zig");
const q8 = @import("../q8/index.zig");

pub const AttentionSpec = struct {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    pub fn validate(self: AttentionSpec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
        if (self.hidden_size != self.num_attention_heads * self.head_dim) return error.HiddenSizeMismatch;
    }

    pub fn queryGroupSize(self: AttentionSpec) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }

    pub fn kvHeadForQueryHead(self: AttentionSpec, q_head_idx: usize) usize {
        return q_head_idx / self.queryGroupSize();
    }
};

pub fn forwardProjectedSingleToken(
    spec: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try basic.scaledDotProductAttentionSingleQuery(
        output,
        projected_query,
        key_cache,
        value_cache,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenBf16Cache(
    spec: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const u16,
    value_cache: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try basic.scaledDotProductAttentionSingleQueryBf16Cache(
        output,
        projected_query,
        key_cache,
        value_cache,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenQ8Cache(
    spec: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try q8.scaledDotProductAttentionSingleQueryQ8Cache(
        output,
        projected_query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenQ8CacheHeadMajor(
    spec: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    data_head_stride: usize,
    scale_head_stride: usize,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try q8.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
        output,
        projected_query,
        key_cache,
        key_scales,
        value_cache,
        value_scales,
        data_head_stride,
        scale_head_stride,
        seq_len,
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

pub fn forwardProjectedSingleTokenQ8CachePagedHeadMajor(
    spec: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
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
    scores_scratch: []f32,
) !void {
    try spec.validate();
    try q8.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor(
        output,
        projected_query,
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
        spec.num_attention_heads,
        spec.num_key_value_heads,
        spec.head_dim,
        scores_scratch,
    );
}

test "gqa attention spec validates grouping and dimensions" {
    const std = @import("std");

    const spec = AttentionSpec{
        .hidden_size = 8,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 2,
    };

    try spec.validate();
    try std.testing.expectEqual(@as(usize, 2), spec.queryGroupSize());
    try std.testing.expectEqual(@as(usize, 0), spec.kvHeadForQueryHead(0));
    try std.testing.expectEqual(@as(usize, 0), spec.kvHeadForQueryHead(1));
    try std.testing.expectEqual(@as(usize, 1), spec.kvHeadForQueryHead(2));
    try std.testing.expectEqual(@as(usize, 1), spec.kvHeadForQueryHead(3));
}
