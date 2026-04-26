const types = @import("ops/types.zig");
const kernels = @import("shared_ops").kernels;
const shared_conv = kernels.conv.entry;

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

fn toSharedOptions(options: Conv2DOptions) shared_conv.Conv2DOptions {
    return .{
        .stride_h = options.stride_h,
        .stride_w = options.stride_w,
        .pad_h = options.pad_h,
        .pad_w = options.pad_w,
        .groups = options.groups,
        .apply_silu = options.apply_silu,
    };
}

fn mapSharedError(err: anyerror) OpError {
    return switch (err) {
        error.ShapeMismatch => OpError.ShapeMismatch,
        error.InvalidOutputShape => OpError.InvalidOutputShape,
        error.InvalidGroups => OpError.InvalidGroups,
        error.InvalidTensorRank => OpError.InvalidTensorRank,
        else => OpError.ShapeMismatch,
    };
}

fn mapLayoutError(err: anyerror) OpError {
    return switch (err) {
        error.ShapeMismatch => OpError.ShapeMismatch,
        error.InvalidOutputShape => OpError.InvalidOutputShape,
        else => OpError.ShapeMismatch,
    };
}

pub fn upsampleNearest(
    input: *const Tensor,
    output: *Tensor,
    scale_h: usize,
    scale_w: usize,
) OpError!void {
    return kernels.layout.upsampleNearestNchw(
        input.data,
        input.shape,
        output.data,
        output.shape,
        scale_h,
        scale_w,
    ) catch |err| return mapLayoutError(err);
}

pub fn concatChannels(inputs: []const *const Tensor, output: *Tensor) OpError!void {
    var views_buffer: [16]kernels.layout.TensorView = undefined;
    if (inputs.len > views_buffer.len) return concatChannelsHeap(inputs, output);

    const views = views_buffer[0..inputs.len];
    for (inputs, views) |input, *view| {
        view.* = .{ .data = input.data, .shape = input.shape };
    }
    return kernels.layout.concatChannelsNchw(views, output.data, output.shape) catch |err| return mapLayoutError(err);
}

fn concatChannelsHeap(inputs: []const *const Tensor, output: *Tensor) OpError!void {
    const views = output.allocator.alloc(kernels.layout.TensorView, inputs.len) catch return OpError.ShapeMismatch;
    defer output.allocator.free(views);
    for (inputs, views) |input, *view| {
        view.* = .{ .data = input.data, .shape = input.shape };
    }
    return kernels.layout.concatChannelsNchw(views, output.data, output.shape) catch |err| return mapLayoutError(err);
}

pub fn copyChannelRange(
    input: *const Tensor,
    input_channel_start: usize,
    channel_count: usize,
    output: *Tensor,
    output_channel_start: usize,
) OpError!void {
    return kernels.layout.copyChannelRangeNchw(
        input.data,
        input.shape,
        input_channel_start,
        channel_count,
        output.data,
        output.shape,
        output_channel_start,
    ) catch |err| return mapLayoutError(err);
}

pub fn copyTensorBlock(
    input: *const Tensor,
    output: *Tensor,
    output_channel_start: usize,
) OpError!void {
    return kernels.layout.copyTensorBlockNchw(
        input.data,
        input.shape,
        output.data,
        output.shape,
        output_channel_start,
    ) catch |err| return mapLayoutError(err);
}

pub fn conv2d(
    input: *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    options: Conv2DOptions,
) OpError!void {
    return shared_conv.conv2d(input, weights, bias, output, toSharedOptions(options)) catch |err| return mapSharedError(err);
}

pub fn conv2dPointwiseConcat(
    inputs: []const *const Tensor,
    weights: *const Tensor,
    bias: ?[]const f32,
    output: *Tensor,
    apply_silu: bool,
) OpError!void {
    return shared_conv.conv2dPointwiseConcat(inputs, weights, bias, output, apply_silu) catch |err| return mapSharedError(err);
}

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
