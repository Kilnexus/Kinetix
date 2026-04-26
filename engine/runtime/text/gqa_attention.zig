const forward = @import("gqa_attention/forward.zig");
const spec = @import("gqa_attention/spec.zig");
const shared_gqa = @import("shared_ops").kernels.attention.gqa;

pub const AttentionSpec = spec.AttentionSpec;

pub const applyRoPEToProjectedHeadsInPlace = forward.applyRoPEToProjectedHeadsInPlace;
pub const applyRoPEToProjectedHeadsWithTableInPlace = forward.applyRoPEToProjectedHeadsWithTableInPlace;
pub const applyRoPEToProjectedHeadsWithPositionInPlace = forward.applyRoPEToProjectedHeadsWithPositionInPlace;

fn toSharedSpec(value: AttentionSpec) shared_gqa.AttentionSpec {
    return .{
        .hidden_size = value.hidden_size,
        .num_attention_heads = value.num_attention_heads,
        .num_key_value_heads = value.num_key_value_heads,
        .head_dim = value.head_dim,
    };
}

pub fn forwardProjectedSingleToken(
    value: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const f32,
    value_cache: []const f32,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try shared_gqa.forwardProjectedSingleToken(toSharedSpec(value), output, projected_query, key_cache, value_cache, seq_len, scores_scratch);
}

pub fn forwardProjectedSingleTokenBf16Cache(
    value: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const u16,
    value_cache: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try shared_gqa.forwardProjectedSingleTokenBf16Cache(toSharedSpec(value), output, projected_query, key_cache, value_cache, seq_len, scores_scratch);
}

pub fn forwardProjectedSingleTokenQ8Cache(
    value: AttentionSpec,
    output: []f32,
    projected_query: []const f32,
    key_cache: []const i8,
    key_scales: []const u16,
    value_cache: []const i8,
    value_scales: []const u16,
    seq_len: usize,
    scores_scratch: []f32,
) !void {
    try shared_gqa.forwardProjectedSingleTokenQ8Cache(toSharedSpec(value), output, projected_query, key_cache, key_scales, value_cache, value_scales, seq_len, scores_scratch);
}

pub fn forwardProjectedSingleTokenQ8CacheHeadMajor(
    value: AttentionSpec,
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
    try shared_gqa.forwardProjectedSingleTokenQ8CacheHeadMajor(toSharedSpec(value), output, projected_query, key_cache, key_scales, value_cache, value_scales, data_head_stride, scale_head_stride, seq_len, scores_scratch);
}

pub fn forwardProjectedSingleTokenQ8CachePagedHeadMajor(
    value: AttentionSpec,
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
    try shared_gqa.forwardProjectedSingleTokenQ8CachePagedHeadMajor(toSharedSpec(value), output, projected_query, key_cache, key_scales, value_cache, value_scales, head_data_stride, head_scale_stride, page_data_stride, page_scale_stride, page_len, pages_per_head, seq_len, scores_scratch);
}
