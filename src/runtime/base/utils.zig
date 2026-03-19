const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const types = @import("types.zig");

pub const Activation = types.Activation;
pub const RuntimeError = types.RuntimeError;
pub const Tensor = types.Tensor;

pub fn sliceChannels(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    channel_start: usize,
    channel_count: usize,
) !Tensor {
    if (channel_start + channel_count > input.shape[1]) return ops.OpError.ShapeMismatch;

    var output = try Tensor.init(
        allocator,
        input.shape[0],
        channel_count,
        input.shape[2],
        input.shape[3],
    );

    const plane = input.shape[2] * input.shape[3];
    const input_batch_stride = input.shape[1] * plane;
    const output_batch_stride = channel_count * plane;

    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input_batch_stride;
        const output_batch_base = n * output_batch_stride;
        for (0..channel_count) |c| {
            const src = input.data[input_batch_base + (channel_start + c) * plane ..][0..plane];
            const dst = output.data[output_batch_base + c * plane ..][0..plane];
            @memcpy(dst, src);
        }
    }
    return output;
}

pub fn sliceChannelsViewBatch1(
    input: *const Tensor,
    channel_start: usize,
    channel_count: usize,
) ops.OpError!Tensor {
    if (input.shape[0] != 1) return ops.OpError.ShapeMismatch;
    if (channel_start + channel_count > input.shape[1]) return ops.OpError.ShapeMismatch;

    const plane = input.shape[2] * input.shape[3];
    const start = channel_start * plane;
    const len = channel_count * plane;
    return .{
        .allocator = undefined,
        .data = input.data[start..][0..len],
        .shape = .{ 1, channel_count, input.shape[2], input.shape[3] },
    };
}

pub fn tensorView(meta: *const graph.TensorMeta, data: []const f32) Tensor {
    return .{
        .allocator = undefined,
        .data = @constCast(data),
        .shape = meta.shape,
    };
}

pub fn applyActivation(output: *Tensor, activation: Activation) void {
    switch (activation) {
        .identity => {},
        .silu => ops.siluInPlace(output),
    }
}

pub fn childModulePath(buffer: []u8, parent: []const u8, child: []const u8) RuntimeError![]const u8 {
    return std.fmt.bufPrint(buffer, "{s}.{s}", .{ parent, child }) catch return error.BufferTooSmall;
}
