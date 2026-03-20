const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;

pub fn upsampleNearest(
    input: *const Tensor,
    output: *Tensor,
    scale_h: usize,
    scale_w: usize,
) OpError!void {
    if (output.shape[0] != input.shape[0] or output.shape[1] != input.shape[1]) {
        return OpError.ShapeMismatch;
    }
    if (output.shape[2] != input.shape[2] * scale_h or output.shape[3] != input.shape[3] * scale_w) {
        return OpError.InvalidOutputShape;
    }

    const in_plane = input.shape[2] * input.shape[3];
    const out_plane = output.shape[2] * output.shape[3];

    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input.shape[1] * in_plane;
        const output_batch_base = n * output.shape[1] * out_plane;
        for (0..input.shape[1]) |c| {
            const input_channel = input.data[input_batch_base + c * in_plane ..][0..in_plane];
            const output_channel = output.data[output_batch_base + c * out_plane ..][0..out_plane];
            for (0..input.shape[2]) |iy| {
                const input_row = input_channel[iy * input.shape[3] ..][0..input.shape[3]];
                for (0..scale_h) |dy| {
                    const output_row = output_channel[(iy * scale_h + dy) * output.shape[3] ..][0..output.shape[3]];
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

pub fn concatChannels(inputs: []const *const Tensor, output: *Tensor) OpError!void {
    if (inputs.len == 0) return OpError.ShapeMismatch;

    const batch = inputs[0].shape[0];
    const height = inputs[0].shape[2];
    const width = inputs[0].shape[3];
    var total_channels: usize = 0;

    for (inputs) |input| {
        if (input.shape[0] != batch or input.shape[2] != height or input.shape[3] != width) {
            return OpError.ShapeMismatch;
        }
        total_channels += input.shape[1];
    }

    if (output.shape[0] != batch or output.shape[1] != total_channels or output.shape[2] != height or output.shape[3] != width) {
        return OpError.InvalidOutputShape;
    }

    var channel_offset: usize = 0;
    for (inputs) |input| {
        const plane = height * width;
        const block_len = input.shape[1] * plane;
        for (0..batch) |n| {
            const input_batch_base = n * input.shape[1] * plane;
            const output_batch_base = n * output.shape[1] * plane + channel_offset * plane;
            const src = input.data[input_batch_base..][0..block_len];
            const dst = output.data[output_batch_base..][0..block_len];
            @memcpy(dst, src);
        }
        channel_offset += input.shape[1];
    }
}

pub fn copyChannelRange(
    input: *const Tensor,
    input_channel_start: usize,
    channel_count: usize,
    output: *Tensor,
    output_channel_start: usize,
) OpError!void {
    if (input.shape[0] != output.shape[0] or input.shape[2] != output.shape[2] or input.shape[3] != output.shape[3]) {
        return OpError.ShapeMismatch;
    }
    if (input_channel_start + channel_count > input.shape[1] or output_channel_start + channel_count > output.shape[1]) {
        return OpError.InvalidOutputShape;
    }

    const plane = input.shape[2] * input.shape[3];
    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input.shape[1] * plane;
        const output_batch_base = n * output.shape[1] * plane;
        for (0..channel_count) |c| {
            const src = input.data[input_batch_base + (input_channel_start + c) * plane ..][0..plane];
            const dst = output.data[output_batch_base + (output_channel_start + c) * plane ..][0..plane];
            @memcpy(dst, src);
        }
    }
}

test "concat channels preserves order" {
    const testing = @import("std").testing;

    var lhs = try Tensor.init(testing.allocator, 1, 1, 1, 2);
    defer lhs.deinit();
    lhs.data[0] = 1.0;
    lhs.data[1] = 2.0;

    var rhs = try Tensor.init(testing.allocator, 1, 2, 1, 2);
    defer rhs.deinit();
    rhs.data[0] = 3.0;
    rhs.data[1] = 4.0;
    rhs.data[2] = 5.0;
    rhs.data[3] = 6.0;

    var output = try Tensor.init(testing.allocator, 1, 3, 1, 2);
    defer output.deinit();

    const inputs = [_]*const Tensor{ &lhs, &rhs };
    try concatChannels(&inputs, &output);

    try testing.expectApproxEqAbs(@as(f32, 1.0), output.get(0, 0, 0, 0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), output.get(0, 1, 0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output.get(0, 2, 0, 1), 1e-6);
}

test "upsample nearest duplicates pixels" {
    const testing = @import("std").testing;

    var input = try Tensor.init(testing.allocator, 1, 1, 1, 2);
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 2.0;

    var output = try Tensor.init(testing.allocator, 1, 1, 2, 4);
    defer output.deinit();

    try upsampleNearest(&input, &output, 2, 2);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output.get(0, 0, 0, 1), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), output.get(0, 0, 1, 3), 1e-6);
}
