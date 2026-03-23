const std = @import("std");
const jpeg_types = @import("types.zig");

const ComponentPlane = jpeg_types.ComponentPlane;

pub fn idctBlock(coeffs: *const [64]i32, quant: *const [64]u16) [64]u8 {
    var out = [_]u8{0} ** 64;
    const inv_sqrt2: f64 = 0.7071067811865476;

    for (0..8) |y| {
        for (0..8) |x| {
            var sum: f64 = 0.0;
            for (0..8) |v| {
                const cv = if (v == 0) inv_sqrt2 else 1.0;
                const cos_y = std.math.cos((@as(f64, @floatFromInt(2 * y + 1)) * @as(f64, @floatFromInt(v)) * std.math.pi) / 16.0);
                for (0..8) |u| {
                    const cu = if (u == 0) inv_sqrt2 else 1.0;
                    const cos_x = std.math.cos((@as(f64, @floatFromInt(2 * x + 1)) * @as(f64, @floatFromInt(u)) * std.math.pi) / 16.0);
                    const idx = v * 8 + u;
                    const value = @as(f64, @floatFromInt(coeffs[idx])) * @as(f64, @floatFromInt(quant[idx]));
                    sum += cu * cv * value * cos_x * cos_y;
                }
            }
            out[y * 8 + x] = clampToU8(@as(f32, @floatCast(sum / 4.0 + 128.0)));
        }
    }

    return out;
}

pub fn writeBlock(plane: *ComponentPlane, block_x: usize, block_y: usize, samples: *const [64]u8) !void {
    const start_x = block_x * 8;
    const start_y = block_y * 8;
    if (start_x + 8 > plane.plane_width or start_y + 8 > plane.plane_height) return error.InvalidJpegData;
    for (0..8) |y| {
        const dst_row = (start_y + y) * plane.plane_width + start_x;
        const src_row = y * 8;
        @memcpy(plane.samples[dst_row .. dst_row + 8], samples[src_row .. src_row + 8]);
    }
}

pub fn samplePlane(plane: *const ComponentPlane, x: usize, y: usize, max_h: u8, max_v: u8) u8 {
    const sample_x = @min((x * @as(usize, plane.h)) / @as(usize, max_h), plane.actual_width - 1);
    const sample_y = @min((y * @as(usize, plane.v)) / @as(usize, max_v), plane.actual_height - 1);
    return plane.samples[sample_y * plane.plane_width + sample_x];
}

pub fn divCeil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn clampToU8(value: f32) u8 {
    if (value <= 0) return 0;
    if (value >= 255) return 255;
    return @intFromFloat(@round(value));
}
