const std = @import("std");
const graph = @import("graph");
const weights_mod = @import("weights");
const base_types = @import("../../base/types.zig");

pub const Tensor = base_types.Tensor;
pub const RuntimeError = base_types.RuntimeError;

pub const ModuleRunnerFn = *const fn (
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor;

pub const C3k2Profile = struct {
    cv1_ns: u64 = 0,
    child_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv2_ns: u64 = 0,
    child_kind: []const u8 = "",
    child_c3k: ?C3kProfile = null,
    child_bottleneck: ?BottleneckProfile = null,
};

pub const C3kProfile = struct {
    cv1_ns: u64 = 0,
    seq_ns: u64 = 0,
    cv2_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv3_ns: u64 = 0,
    seq_kind: []const u8 = "",
};

pub const BottleneckProfile = struct {
    cv1_ns: u64 = 0,
    cv2_ns: u64 = 0,
    add_ns: u64 = 0,
    has_add: bool = false,
};

pub const SPPFProfile = struct {
    cv1_ns: u64 = 0,
    pool1_ns: u64 = 0,
    pool2_ns: u64 = 0,
    pool3_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv2_ns: u64 = 0,
};

pub const ProfiledTensor = struct {
    output: Tensor,
    c3k2_profile: C3k2Profile,
};

pub const BottleneckProfiledTensor = struct {
    output: Tensor,
    bottleneck_profile: BottleneckProfile,
};

pub const SPPFProfiledTensor = struct {
    output: Tensor,
    sppf_profile: SPPFProfile,
};

pub const C3kProfiledTensor = struct {
    output: Tensor,
    c3k_profile: C3kProfile,
};
