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

    for (0..input.shape[0]) |n| {
        for (0..channel_count) |c| {
            for (0..input.shape[2]) |y| {
                for (0..input.shape[3]) |x| {
                    output.set(n, c, y, x, input.get(n, channel_start + c, y, x));
                }
            }
        }
    }
    return output;
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
