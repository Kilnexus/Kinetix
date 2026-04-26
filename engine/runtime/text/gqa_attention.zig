const decoder_types = @import("decoder_types.zig");
const spec = @import("gqa_attention/spec.zig");
const shared_attention = @import("shared_ops").kernels.attention;
const shared_gqa = shared_attention.gqa;
const shared_rope = shared_attention.rope;

pub const AttentionSpec = spec.AttentionSpec;

pub fn applyRoPEToProjectedHeadsInPlace(
    value: AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    position: usize,
) !void {
    return shared_rope.applyRoPEToProjectedHeadsInPlace(
        toSharedRoPESpec(value),
        projected_query,
        projected_key,
        position,
    );
}

pub fn applyRoPEToProjectedHeadsWithTableInPlace(
    value: AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const shared_rope.RoPETable,
    position: usize,
) !void {
    return shared_rope.applyRoPEToProjectedHeadsWithTableInPlace(
        toSharedRoPESpec(value),
        projected_query,
        projected_key,
        table,
        position,
    );
}

pub fn applyRoPEToProjectedHeadsWithPositionInPlace(
    value: AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const shared_rope.RoPETable,
    position: decoder_types.TokenPosition,
) !void {
    return shared_rope.applyRoPEToProjectedHeadsWithPositionInPlace(
        toSharedRoPESpec(value),
        projected_query,
        projected_key,
        table,
        toSharedPosition(position),
    );
}

fn toSharedSpec(value: AttentionSpec) shared_gqa.AttentionSpec {
    return .{
        .hidden_size = value.hidden_size,
        .num_attention_heads = value.num_attention_heads,
        .num_key_value_heads = value.num_key_value_heads,
        .head_dim = value.head_dim,
    };
}

fn toSharedRoPESpec(value: AttentionSpec) shared_rope.ProjectedHeadsSpec {
    return .{
        .hidden_size = value.hidden_size,
        .num_attention_heads = value.num_attention_heads,
        .num_key_value_heads = value.num_key_value_heads,
        .head_dim = value.head_dim,
        .rope_theta = value.rope_theta,
        .rope_position_mode = toSharedPositionMode(value.rope_position_mode),
        .mrope_sections = value.mrope_sections,
    };
}

fn toSharedPosition(position: decoder_types.TokenPosition) shared_rope.Position {
    return switch (position.mode) {
        .scalar => shared_rope.Position.scalarPosition(position.scalar),
        .mrope => shared_rope.Position.mropePosition(position.axes),
    };
}

fn toSharedPositionMode(mode: decoder_types.RopePositionMode) shared_rope.PositionMode {
    return switch (mode) {
        .scalar => .scalar,
        .mrope => .mrope,
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
