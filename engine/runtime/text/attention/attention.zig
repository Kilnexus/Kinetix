const decoder_types = @import("../decoder_types.zig");
const shared_attention = @import("shared_ops").kernels.attention;
const basic = shared_attention.basic;
const q8 = shared_attention.q8;
const rope = shared_attention.rope;

pub const RoPETable = rope.RoPETable;

pub const applyRoPEToHeadInPlace = rope.applyRoPEToHeadInPlace;
pub const applyRoPEToHeadWithTableInPlace = rope.applyRoPEToHeadWithTableInPlace;
pub const applyRoPEToHeadsInPlace = rope.applyRoPEToHeadsInPlace;
pub const applyRoPEToHeadsWithTableInPlace = rope.applyRoPEToHeadsWithTableInPlace;

pub const softmaxInPlace = basic.softmaxInPlace;
pub const scaledDotProductAttentionSingleQuery = basic.scaledDotProductAttentionSingleQuery;
pub const scaledDotProductAttentionSingleQueryBf16Cache = basic.scaledDotProductAttentionSingleQueryBf16Cache;

pub const q8_cache_group_size = q8.q8_cache_group_size;
pub const scaledDotProductAttentionSingleQueryQ8Cache = q8.scaledDotProductAttentionSingleQueryQ8Cache;
pub const scaledDotProductAttentionSingleQueryQ8CacheHeadMajor = q8.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor;
pub const scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor = q8.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor;
pub const dotQ8GroupedSlice = q8.dotQ8GroupedSlice;
pub const axpyQ8GroupedSliceInPlace = q8.axpyQ8GroupedSliceInPlace;

pub fn applyRoPEToHeadWithPositionInPlace(
    head: []f32,
    table: *const RoPETable,
    position: decoder_types.TokenPosition,
    mrope_sections: [4]u32,
) !void {
    return rope.applyRoPEToHeadWithPositionInPlace(
        head,
        table,
        toSharedPosition(position),
        mrope_sections,
    );
}

pub fn applyRoPEToHeadsWithPositionInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    table: *const RoPETable,
    position: decoder_types.TokenPosition,
    mrope_sections: [4]u32,
) !void {
    return rope.applyRoPEToHeadsWithPositionInPlace(
        heads,
        num_heads,
        head_dim,
        table,
        toSharedPosition(position),
        mrope_sections,
    );
}

fn toSharedPosition(position: decoder_types.TokenPosition) rope.Position {
    return switch (position.mode) {
        .scalar => rope.Position.scalarPosition(position.scalar),
        .mrope => rope.Position.mropePosition(position.axes),
    };
}
