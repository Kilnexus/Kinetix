const std = @import("std");
const legacy = @import("../../../legacy/zinfer/src/model/runtime/optimized_kv_cache.zig");

pub const Scheme = legacy.Scheme;
pub const resolveScheme = legacy.resolveScheme;
pub const q8_group_size = legacy.q8_group_size;
pub const q8_page_len = legacy.q8_page_len;
pub const Q8Layout = legacy.Q8Layout;
pub const default_q8_layout = legacy.default_q8_layout;

pub const LayerKVCache = legacy.LayerKVCache;
pub const ModelCache = legacy.ModelCache;
pub const estimateBytes = legacy.estimateBytes;
pub const estimateBytesWithLayout = legacy.estimateBytesWithLayout;

pub const quantizeQ8Slice = legacy.quantizeQ8Slice;
pub const scaleGroupsPerToken = legacy.scaleGroupsPerToken;

test "bridge preserves default q8 layout" {
    try std.testing.expectEqual(default_q8_layout, legacy.default_q8_layout);
}
