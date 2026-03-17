const std = @import("std");
const types = @import("types.zig");

pub const ImageU8 = types.ImageU8;
pub const ImageError = types.ImageError;

pub fn resizeBilinear(
    allocator: std.mem.Allocator,
    src: *const ImageU8,
    target_width: usize,
    target_height: usize,
) !ImageU8 {
    if (src.channels == 0) return error.InvalidChannelCount;

    var dst = try ImageU8.init(allocator, target_width, target_height, src.channels);
    errdefer dst.deinit();

    for (0..target_height) |dy| {
        const src_y = ((@as(f32, @floatFromInt(dy)) + 0.5) * @as(f32, @floatFromInt(src.height)) / @as(f32, @floatFromInt(target_height))) - 0.5;
        const y0 = clampIndex(@as(isize, @intFromFloat(@floor(src_y))), src.height);
        const y1 = if (y0 + 1 < src.height) y0 + 1 else src.height - 1;
        const wy = src_y - @as(f32, @floatFromInt(y0));

        for (0..target_width) |dx| {
            const src_x = ((@as(f32, @floatFromInt(dx)) + 0.5) * @as(f32, @floatFromInt(src.width)) / @as(f32, @floatFromInt(target_width))) - 0.5;
            const x0 = clampIndex(@as(isize, @intFromFloat(@floor(src_x))), src.width);
            const x1 = if (x0 + 1 < src.width) x0 + 1 else src.width - 1;
            const wx = src_x - @as(f32, @floatFromInt(x0));

            for (0..src.channels) |channel| {
                const p00 = @as(f32, @floatFromInt(src.get(x0, y0, channel)));
                const p10 = @as(f32, @floatFromInt(src.get(x1, y0, channel)));
                const p01 = @as(f32, @floatFromInt(src.get(x0, y1, channel)));
                const p11 = @as(f32, @floatFromInt(src.get(x1, y1, channel)));

                const top = lerp(p00, p10, wx);
                const bottom = lerp(p01, p11, wx);
                const value = lerp(top, bottom, wy);
                dst.set(dx, dy, channel, @intFromFloat(@round(value)));
            }
        }
    }

    return dst;
}

fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn clampIndex(value: isize, upper: usize) usize {
    if (value < 0) return 0;
    const upper_index: isize = @intCast(upper - 1);
    if (value > upper_index) return upper - 1;
    return @intCast(value);
}
