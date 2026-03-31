const common = @import("common.zig");

pub const Conv2DTask = struct {
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
    oc_start: usize,
    oc_end: usize,
};

pub const Conv2DPointwiseTask = struct {
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    groups: usize,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
};

pub const Conv2DPointwiseConcatTask = struct {
    inputs: []const *const common.Tensor,
    input_channel_offsets: []const usize,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    oc_start: usize,
    oc_end: usize,
    apply_silu: bool,
};
