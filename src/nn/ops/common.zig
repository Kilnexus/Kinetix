const std = @import("std");
const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;

pub fn siluInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = x / (1.0 + @exp(-x));
    }
}

pub fn sigmoidInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = 1.0 / (1.0 + @exp(-x));
    }
}

pub fn add(output: *Tensor, lhs: *const Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(lhs) or !lhs.sameShape(rhs)) return OpError.ShapeMismatch;
    for (output.data, lhs.data, rhs.data) |*out, left, right| out.* = left + right;
}

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
        for (0..batch) |n| {
            const input_batch_base = n * input.shape[1] * plane;
            const output_batch_base = n * output.shape[1] * plane;
            for (0..input.shape[1]) |c| {
                const src = input.data[input_batch_base + c * plane ..][0..plane];
                const dst = output.data[output_batch_base + (channel_offset + c) * plane ..][0..plane];
                @memcpy(dst, src);
            }
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
    if (input.shape[0] != output.shape[0] or input.shape[1] != output.shape[1]) {
        return OpError.ShapeMismatch;
    }

    const expected_h = ((input.shape[2] + 2 * pad_h - kernel_h) / stride_h) + 1;
    const expected_w = ((input.shape[3] + 2 * pad_w - kernel_w) / stride_w) + 1;
    if (output.shape[2] != expected_h or output.shape[3] != expected_w) {
        return OpError.InvalidOutputShape;
    }

    for (0..input.shape[0]) |n| {
        for (0..input.shape[1]) |c| {
            for (0..output.shape[2]) |oy| {
                for (0..output.shape[3]) |ox| {
                    var max_value = -std.math.inf(f32);
                    const base_y = @as(isize, @intCast(oy * stride_h)) - @as(isize, @intCast(pad_h));
                    const base_x = @as(isize, @intCast(ox * stride_w)) - @as(isize, @intCast(pad_w));

                    for (0..kernel_h) |ky| {
                        const in_y = base_y + @as(isize, @intCast(ky));
                        if (in_y < 0 or in_y >= @as(isize, @intCast(input.shape[2]))) continue;

                        for (0..kernel_w) |kx| {
                            const in_x = base_x + @as(isize, @intCast(kx));
                            if (in_x < 0 or in_x >= @as(isize, @intCast(input.shape[3]))) continue;

                            const value = input.get(n, c, @intCast(in_y), @intCast(in_x));
                            if (value > max_value) max_value = value;
                        }
                    }
                    output.set(n, c, oy, ox, max_value);
                }
            }
        }
    }
}

pub fn matmul(
    lhs: []const f32,
    rhs: []const f32,
    out: []f32,
    rows: usize,
    shared: usize,
    cols: usize,
) OpError!void {
    if (lhs.len != rows * shared or rhs.len != shared * cols or out.len != rows * cols) {
        return OpError.ShapeMismatch;
    }

    for (0..rows) |r| {
        for (0..cols) |c| {
            var acc: f32 = 0.0;
            for (0..shared) |k| {
                acc += lhs[r * shared + k] * rhs[k * cols + c];
            }
            out[r * cols + c] = acc;
        }
    }
}

pub fn softmaxRows(data: []f32, rows: usize, cols: usize) OpError!void {
    if (data.len != rows * cols) return OpError.ShapeMismatch;

    for (0..rows) |r| {
        const row = data[r * cols .. (r + 1) * cols];
        var max_value = row[0];
        for (row[1..]) |value| {
            if (value > max_value) max_value = value;
        }

        var sum: f32 = 0.0;
        for (row) |*value| {
            value.* = @exp(value.* - max_value);
            sum += value.*;
        }
        for (row) |*value| value.* /= sum;
    }
}

test "concat channels preserves order" {
    const testing = std.testing;

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
    const testing = std.testing;

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

test "softmax row sums to one" {
    const testing = std.testing;
    var values = [_]f32{ 1.0, 2.0, 3.0, 1.0, 1.0, 1.0 };
    try softmaxRows(&values, 2, 3);

    try testing.expectApproxEqAbs(@as(f32, 1.0), values[0] + values[1] + values[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), values[3] + values[4] + values[5], 1e-5);
    try testing.expect(values[2] > values[1]);
    try testing.expect(values[1] > values[0]);
}
