const std = @import("std");

pub const Error = error{
    ShapeMismatch,
    InvalidOutputShape,
};

pub fn maxPool2dNchw(
    input_data: []const f32,
    input_shape: [4]usize,
    output_data: []f32,
    output_shape: [4]usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) Error!void {
    if (input_shape[0] != output_shape[0] or input_shape[1] != output_shape[1]) {
        return error.ShapeMismatch;
    }
    const expected_h = ((input_shape[2] + 2 * pad_h - kernel_h) / stride_h) + 1;
    const expected_w = ((input_shape[3] + 2 * pad_w - kernel_w) / stride_w) + 1;
    if (output_shape[2] != expected_h or output_shape[3] != expected_w) {
        return error.InvalidOutputShape;
    }
    if (input_data.len != input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]) return error.ShapeMismatch;
    if (output_data.len != output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) return error.ShapeMismatch;

    const in_height = input_shape[2];
    const in_width = input_shape[3];
    const in_plane = in_height * in_width;
    const out_width = output_shape[3];
    const out_plane = output_shape[2] * out_width;

    for (0..input_shape[0]) |n| {
        const input_batch_base = n * input_shape[1] * in_plane;
        const output_batch_base = n * output_shape[1] * out_plane;
        for (0..input_shape[1]) |c| {
            const input_channel = input_data[input_batch_base + c * in_plane ..][0..in_plane];
            const output_channel = output_data[output_batch_base + c * out_plane ..][0..out_plane];
            for (0..output_shape[2]) |oy| {
                const base_y = @as(isize, @intCast(oy * stride_h)) - @as(isize, @intCast(pad_h));
                const out_row = output_channel[oy * out_width ..][0..out_width];
                for (0..out_width) |ox| {
                    var max_value = -std.math.inf(f32);
                    const base_x = @as(isize, @intCast(ox * stride_w)) - @as(isize, @intCast(pad_w));

                    for (0..kernel_h) |ky| {
                        const in_y = base_y + @as(isize, @intCast(ky));
                        if (in_y < 0 or in_y >= @as(isize, @intCast(in_height))) continue;

                        const input_row = input_channel[@as(usize, @intCast(in_y)) * in_width ..][0..in_width];
                        for (0..kernel_w) |kx| {
                            const in_x = base_x + @as(isize, @intCast(kx));
                            if (in_x < 0 or in_x >= @as(isize, @intCast(in_width))) continue;

                            const value = input_row[@as(usize, @intCast(in_x))];
                            if (value > max_value) max_value = value;
                        }
                    }
                    out_row[ox] = max_value;
                }
            }
        }
    }
}

test "kernel pooling maxpool selects local maximum" {
    const input = [_]f32{ 1.0, 9.0, 3.0, 4.0 };
    var output = [_]f32{0.0};

    try maxPool2dNchw(&input, .{ 1, 1, 2, 2 }, &output, .{ 1, 1, 1, 1 }, 2, 2, 2, 2, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), output[0], 1e-6);
}
