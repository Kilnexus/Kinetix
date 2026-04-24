const std = @import("std");

pub const Error = error{
    ShapeMismatch,
    InvalidOutputShape,
};

pub const TensorView = struct {
    data: []const f32,
    shape: [4]usize,
};

pub fn upsampleNearestNchw(
    input_data: []const f32,
    input_shape: [4]usize,
    output_data: []f32,
    output_shape: [4]usize,
    scale_h: usize,
    scale_w: usize,
) Error!void {
    if (output_shape[0] != input_shape[0] or output_shape[1] != input_shape[1]) {
        return error.ShapeMismatch;
    }
    if (output_shape[2] != input_shape[2] * scale_h or output_shape[3] != input_shape[3] * scale_w) {
        return error.InvalidOutputShape;
    }
    if (input_data.len != elementCount(input_shape) or output_data.len != elementCount(output_shape)) {
        return error.ShapeMismatch;
    }

    const in_plane = input_shape[2] * input_shape[3];
    const out_plane = output_shape[2] * output_shape[3];

    for (0..input_shape[0]) |n| {
        const input_batch_base = n * input_shape[1] * in_plane;
        const output_batch_base = n * output_shape[1] * out_plane;
        for (0..input_shape[1]) |c| {
            const input_channel = input_data[input_batch_base + c * in_plane ..][0..in_plane];
            const output_channel = output_data[output_batch_base + c * out_plane ..][0..out_plane];
            for (0..input_shape[2]) |iy| {
                const input_row = input_channel[iy * input_shape[3] ..][0..input_shape[3]];
                for (0..scale_h) |dy| {
                    const output_row = output_channel[(iy * scale_h + dy) * output_shape[3] ..][0..output_shape[3]];
                    for (input_row, 0..) |value, ix| {
                        const out_x = ix * scale_w;
                        for (0..scale_w) |dx| {
                            output_row[out_x + dx] = value;
                        }
                    }
                }
            }
        }
    }
}

pub fn concatChannelsNchw(inputs: []const TensorView, output_data: []f32, output_shape: [4]usize) Error!void {
    if (inputs.len == 0) return error.ShapeMismatch;

    const batch = inputs[0].shape[0];
    const height = inputs[0].shape[2];
    const width = inputs[0].shape[3];
    var total_channels: usize = 0;

    for (inputs) |input| {
        if (input.shape[0] != batch or input.shape[2] != height or input.shape[3] != width) {
            return error.ShapeMismatch;
        }
        if (input.data.len != elementCount(input.shape)) return error.ShapeMismatch;
        total_channels += input.shape[1];
    }

    if (output_shape[0] != batch or output_shape[1] != total_channels or output_shape[2] != height or output_shape[3] != width) {
        return error.InvalidOutputShape;
    }
    if (output_data.len != elementCount(output_shape)) return error.ShapeMismatch;

    var channel_offset: usize = 0;
    for (inputs) |input| {
        const plane = height * width;
        const block_len = input.shape[1] * plane;
        for (0..batch) |n| {
            const input_batch_base = n * input.shape[1] * plane;
            const output_batch_base = n * output_shape[1] * plane + channel_offset * plane;
            const src = input.data[input_batch_base..][0..block_len];
            const dst = output_data[output_batch_base..][0..block_len];
            @memcpy(dst, src);
        }
        channel_offset += input.shape[1];
    }
}

pub fn copyChannelRangeNchw(
    input_data: []const f32,
    input_shape: [4]usize,
    input_channel_start: usize,
    channel_count: usize,
    output_data: []f32,
    output_shape: [4]usize,
    output_channel_start: usize,
) Error!void {
    if (input_shape[0] != output_shape[0] or input_shape[2] != output_shape[2] or input_shape[3] != output_shape[3]) {
        return error.ShapeMismatch;
    }
    if (input_channel_start + channel_count > input_shape[1] or output_channel_start + channel_count > output_shape[1]) {
        return error.InvalidOutputShape;
    }
    if (input_data.len != elementCount(input_shape) or output_data.len != elementCount(output_shape)) {
        return error.ShapeMismatch;
    }

    const plane = input_shape[2] * input_shape[3];
    for (0..input_shape[0]) |n| {
        const input_batch_base = n * input_shape[1] * plane;
        const output_batch_base = n * output_shape[1] * plane;
        for (0..channel_count) |c| {
            const src = input_data[input_batch_base + (input_channel_start + c) * plane ..][0..plane];
            const dst = output_data[output_batch_base + (output_channel_start + c) * plane ..][0..plane];
            @memcpy(dst, src);
        }
    }
}

pub fn copyTensorBlockNchw(
    input_data: []const f32,
    input_shape: [4]usize,
    output_data: []f32,
    output_shape: [4]usize,
    output_channel_start: usize,
) Error!void {
    if (input_shape[0] != output_shape[0] or input_shape[2] != output_shape[2] or input_shape[3] != output_shape[3]) {
        return error.ShapeMismatch;
    }
    if (output_channel_start + input_shape[1] > output_shape[1]) {
        return error.InvalidOutputShape;
    }
    if (input_data.len != elementCount(input_shape) or output_data.len != elementCount(output_shape)) {
        return error.ShapeMismatch;
    }

    const plane = input_shape[2] * input_shape[3];
    const block_len = input_shape[1] * plane;
    for (0..input_shape[0]) |n| {
        const input_batch_base = n * block_len;
        const output_batch_base = n * output_shape[1] * plane + output_channel_start * plane;
        const src = input_data[input_batch_base..][0..block_len];
        const dst = output_data[output_batch_base..][0..block_len];
        @memcpy(dst, src);
    }
}

fn elementCount(shape: [4]usize) usize {
    return shape[0] * shape[1] * shape[2] * shape[3];
}

test "kernel layout concat channels preserves order" {
    const lhs = [_]f32{ 1.0, 2.0 };
    const rhs = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    var output = [_]f32{0.0} ** 6;

    const inputs = [_]TensorView{
        .{ .data = &lhs, .shape = .{ 1, 1, 1, 2 } },
        .{ .data = &rhs, .shape = .{ 1, 2, 1, 2 } },
    };
    try concatChannelsNchw(&inputs, &output, .{ 1, 3, 1, 2 });

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[5], 1e-6);
}

test "kernel layout upsample nearest duplicates pixels" {
    const input = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{0.0} ** 8;

    try upsampleNearestNchw(&input, .{ 1, 1, 1, 2 }, &output, .{ 1, 1, 2, 4 }, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[7], 1e-6);
}
