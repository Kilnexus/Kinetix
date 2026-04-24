const std = @import("std");
const types = @import("types.zig");
const kernels = @import("shared_ops").kernels;

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;

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

test "maxpool selects local maximum" {
    const testing = std.testing;

    var input = try Tensor.init(testing.allocator, 1, 1, 2, 2);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 9.0;
    input.data[2] = 3.0;
    input.data[3] = 4.0;

    var output = try Tensor.init(testing.allocator, 1, 1, 1, 1);
    defer output.deinit();

    try maxPool2d(&input, &output, 2, 2, 2, 2, 0, 0);
    try testing.expectApproxEqAbs(@as(f32, 9.0), output.data[0], 1e-6);
}
