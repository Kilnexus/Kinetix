const types = @import("ops/types.zig");
const activation = @import("ops/activation.zig");
const layout = @import("ops/layout.zig");
const pooling = @import("ops/pooling.zig");
const linalg = @import("ops/linalg.zig");
const conv = @import("ops/conv.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

pub const siluInPlace = activation.siluInPlace;
pub const sigmoidInPlace = activation.sigmoidInPlace;
pub const add = activation.add;
pub const addInPlace = activation.addInPlace;
pub const upsampleNearest = layout.upsampleNearest;
pub const concatChannels = layout.concatChannels;
pub const copyChannelRange = layout.copyChannelRange;
pub const maxPool2d = pooling.maxPool2d;
pub const matmul = linalg.matmul;
pub const softmaxRows = linalg.softmaxRows;
pub const conv2d = conv.conv2d;

test {
    _ = @import("ops/activation.zig");
    _ = @import("ops/layout.zig");
    _ = @import("ops/pooling.zig");
    _ = @import("ops/linalg.zig");
    _ = @import("ops/conv.zig");
}
