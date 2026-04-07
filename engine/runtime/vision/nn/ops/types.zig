const tensor_mod = @import("tensor");

pub const Tensor = tensor_mod.Tensor;

pub const OpError = error{
    ShapeMismatch,
    InvalidOutputShape,
    InvalidGroups,
    InvalidTensorRank,
};

pub const Conv2DOptions = struct {
    stride_h: usize = 1,
    stride_w: usize = 1,
    pad_h: usize = 0,
    pad_w: usize = 0,
    groups: usize = 1,
    apply_silu: bool = false,
};
