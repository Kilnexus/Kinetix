const std = @import("std");
const types = @import("kv_cache_types.zig");
const legacy_cache = @import("../../../legacy/zinfer/src/model/runtime/optimized_kv_cache/cache.zig");
const legacy_quantize = @import("../../../legacy/zinfer/src/model/runtime/optimized_kv_cache/quantize.zig");

pub const Scheme = types.Scheme;
pub const resolveScheme = types.resolveScheme;
pub const q8_group_size = types.q8_group_size;
pub const q8_page_len = types.q8_page_len;
pub const Q8Layout = types.Q8Layout;
pub const default_q8_layout = types.default_q8_layout;

pub const LayerKVCache = legacy_cache.LayerKVCache;
pub const ModelCache = legacy_cache.ModelCache;
pub const estimateBytes = legacy_cache.estimateBytes;
pub const estimateBytesWithLayout = legacy_cache.estimateBytesWithLayout;

pub const quantizeQ8Slice = legacy_quantize.quantizeQ8Slice;
pub const scaleGroupsPerToken = legacy_quantize.scaleGroupsPerToken;

test "bridge preserves default q8 layout" {
    try std.testing.expectEqual(default_q8_layout, types.default_q8_layout);
}
