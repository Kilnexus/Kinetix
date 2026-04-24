const rope = @import("rope.zig");
const shared_attention = @import("shared_ops").kernels.attention;
const basic = shared_attention.basic;
const q8 = shared_attention.q8;

pub const RoPETable = rope.RoPETable;

pub const applyRoPEToHeadInPlace = rope.applyRoPEToHeadInPlace;
pub const applyRoPEToHeadWithTableInPlace = rope.applyRoPEToHeadWithTableInPlace;
pub const applyRoPEToHeadWithPositionInPlace = rope.applyRoPEToHeadWithPositionInPlace;
pub const applyRoPEToHeadsInPlace = rope.applyRoPEToHeadsInPlace;
pub const applyRoPEToHeadsWithTableInPlace = rope.applyRoPEToHeadsWithTableInPlace;
pub const applyRoPEToHeadsWithPositionInPlace = rope.applyRoPEToHeadsWithPositionInPlace;

pub const softmaxInPlace = basic.softmaxInPlace;
pub const scaledDotProductAttentionSingleQuery = basic.scaledDotProductAttentionSingleQuery;
pub const scaledDotProductAttentionSingleQueryBf16Cache = basic.scaledDotProductAttentionSingleQueryBf16Cache;

pub const q8_cache_group_size = q8.q8_cache_group_size;
pub const scaledDotProductAttentionSingleQueryQ8Cache = q8.scaledDotProductAttentionSingleQueryQ8Cache;
pub const scaledDotProductAttentionSingleQueryQ8CacheHeadMajor = q8.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor;
pub const scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor = q8.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor;
pub const dotQ8GroupedSlice = q8.dotQ8GroupedSlice;
pub const axpyQ8GroupedSliceInPlace = q8.axpyQ8GroupedSliceInPlace;
