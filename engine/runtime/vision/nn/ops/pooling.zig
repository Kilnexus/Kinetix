const std = @import("std");
const types = @import("types.zig");

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
    if (input.shape[0] != output.shape[0] or input.shape[1] != output.shape[1]) {
        return OpError.ShapeMismatch;
    }

    const expected_h = ((input.shape[2] + 2 * pad_h - kernel_h) / stride_h) + 1;
    const expected_w = ((input.shape[3] + 2 * pad_w - kernel_w) / stride_w) + 1;
    if (output.shape[2] != expected_h or output.shape[3] != expected_w) {
        return OpError.InvalidOutputShape;
    }

    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const in_plane = in_height * in_width;
    const out_width = output.shape[3];
    const out_plane = output.shape[2] * out_width;

    for (0..input.shape[0]) |n| {
        const input_batch_base = n * input.shape[1] * in_plane;
        const output_batch_base = n * output.shape[1] * out_plane;
        for (0..input.shape[1]) |c| {
            const input_channel = input.data[input_batch_base + c * in_plane ..][0..in_plane];
            const output_channel = output.data[output_batch_base + c * out_plane ..][0..out_plane];
            for (0..output.shape[2]) |oy| {
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
