const types = @import("ops/types.zig");
const common = @import("ops/common.zig");
const conv = @import("ops/conv.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

pub const siluInPlace = common.siluInPlace;
pub const sigmoidInPlace = common.sigmoidInPlace;
pub const add = common.add;
pub const upsampleNearest = common.upsampleNearest;
pub const concatChannels = common.concatChannels;
pub const copyChannelRange = common.copyChannelRange;
pub const maxPool2d = common.maxPool2d;
pub const matmul = common.matmul;
pub const softmaxRows = common.softmaxRows;
pub const conv2d = conv.conv2d;

test {
    _ = @import("ops/common.zig");
    _ = @import("ops/conv.zig");
}
