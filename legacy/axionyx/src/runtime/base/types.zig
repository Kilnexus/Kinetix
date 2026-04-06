const graph = @import("graph");
const tensor_mod = @import("tensor");

pub const TensorDesc = struct {
    shape: [4]usize,
    len: usize,
};

pub const Tensor = tensor_mod.Tensor;

pub const Activation = enum {
    identity,
    silu,
};

pub const RuntimeError = error{
    BufferTooSmall,
    InvalidAttributeType,
    InvalidModuleKind,
    MissingAttribute,
    ModuleNotFound,
    TensorNotFound,
};

pub const ConvSpec = struct {
    weight: *const graph.TensorMeta,
    bias: ?*const graph.TensorMeta,
    stride: [2]usize,
    padding: [2]usize,
    groups: usize,
    activation: Activation,
};

pub fn shapeLen(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| total *= dim;
    return total;
}
