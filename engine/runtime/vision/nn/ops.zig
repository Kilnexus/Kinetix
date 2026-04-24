const types = @import("ops/types.zig");
const layout = @import("ops/layout.zig");
const conv = @import("ops/conv.zig");
const kernels = @import("shared_ops").kernels;

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

pub const upsampleNearest = layout.upsampleNearest;
pub const concatChannels = layout.concatChannels;
pub const copyChannelRange = layout.copyChannelRange;
pub const copyTensorBlock = layout.copyTensorBlock;
pub const conv2d = conv.conv2d;
pub const conv2dPointwiseConcat = conv.conv2dPointwiseConcat;

pub fn siluInPlace(tensor: *Tensor) void {
    kernels.activation.siluInPlace(tensor.data);
}

pub fn sigmoidInPlace(tensor: *Tensor) void {
    kernels.activation.sigmoidInPlace(tensor.data);
}

pub fn add(output: *Tensor, lhs: *const Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(lhs) or !lhs.sameShape(rhs)) return OpError.ShapeMismatch;
    kernels.activation.add(output.data, lhs.data, rhs.data) catch return OpError.ShapeMismatch;
}

pub fn addInPlace(output: *Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(rhs)) return OpError.ShapeMismatch;
    kernels.activation.addInPlace(output.data, rhs.data) catch return OpError.ShapeMismatch;
}

pub fn addInPlaceUnchecked(output: *Tensor, rhs: *const Tensor) void {
    kernels.activation.addInPlaceUnchecked(output.data, rhs.data);
}

pub fn maxPool2d(
    input: *const Tensor,
    output: *Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) OpError!void {
    kernels.pooling.maxPool2dNchw(
        input.data,
        input.shape,
        output.data,
        output.shape,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    ) catch |err| switch (err) {
        error.ShapeMismatch => return OpError.ShapeMismatch,
        error.InvalidOutputShape => return OpError.InvalidOutputShape,
    };
}

pub fn matmul(
    lhs: []const f32,
    rhs: []const f32,
    out: []f32,
    rows: usize,
    shared: usize,
    cols: usize,
) OpError!void {
    kernels.linalg.matmul(lhs, rhs, out, rows, shared, cols) catch return OpError.ShapeMismatch;
}

pub fn softmaxRows(data: []f32, rows: usize, cols: usize) OpError!void {
    kernels.linalg.softmaxRows(data, rows, cols) catch return OpError.ShapeMismatch;
}

test {
    _ = @import("ops/layout.zig");
    _ = @import("ops/conv.zig");
}
