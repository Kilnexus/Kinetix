const std = @import("std");
const types = @import("types.zig");
const color_cache = @import("color_cache.zig");

pub const Vp8lArgbImage = types.Vp8lArgbImage;
pub const Vp8lTransform = types.Vp8lTransform;

pub fn applySupportedTransformsInPlace(image: *Vp8lArgbImage, transforms: []const Vp8lTransform) !void {
    var i = transforms.len;
    while (i > 0) {
        i -= 1;
        switch (transforms[i].kind) {
            .subtract_green => applySubtractGreenTransformInPlace(image.pixels),
            else => return error.UnsupportedWebpBitstream,
        }
    }
}

pub fn applySubtractGreenTransformInPlace(pixels: []u32) void {
    for (pixels) |*pixel_ptr| {
        const pixel = pixel_ptr.*;
        const alpha = (pixel >> 24) & 0xff;
        const red_delta = (pixel >> 16) & 0xff;
        const green = (pixel >> 8) & 0xff;
        const blue_delta = pixel & 0xff;
        const red = (red_delta + green) & 0xff;
        const blue = (blue_delta + green) & 0xff;
        pixel_ptr.* = color_cache.packArgb(
            @intCast(alpha),
            @intCast(red),
            @intCast(green),
            @intCast(blue),
        );
    }
}

pub fn restoreColorIndexPaletteInPlace(pixels: []u32) void {
    var prev = color_cache.packArgb(0, 0, 0, 0);
    for (pixels) |*pixel_ptr| {
        const pixel = pixel_ptr.*;
        const restored = color_cache.packArgb(
            @intCast((((pixel >> 24) & 0xff) + ((prev >> 24) & 0xff)) & 0xff),
            @intCast((((pixel >> 16) & 0xff) + ((prev >> 16) & 0xff)) & 0xff),
            @intCast((((pixel >> 8) & 0xff) + ((prev >> 8) & 0xff)) & 0xff),
            @intCast(((pixel & 0xff) + (prev & 0xff)) & 0xff),
        );
        pixel_ptr.* = restored;
        prev = restored;
    }
}

pub fn expandColorIndexedImage(
    allocator: std.mem.Allocator,
    indexed_pixels: []const u32,
    palette: []const u32,
    width_bits: usize,
    output_width: usize,
    output_height: usize,
    end_bit_pos: usize,
) !Vp8lArgbImage {
    const pixels_per_index_byte = @as(usize, 1) << @intCast(width_bits);
    const bits_per_index = 8 / pixels_per_index_byte;
    const index_mask = (@as(usize, 1) << @intCast(bits_per_index)) - 1;
    const output_len = output_width * output_height;

    const pixels = try allocator.alloc(u32, output_len);
    errdefer allocator.free(pixels);

    var written: usize = 0;
    for (indexed_pixels) |pixel| {
        const packed_green = @as(usize, (pixel >> 8) & 0xff);
        for (0..pixels_per_index_byte) |slot| {
            if (written >= output_len) break;
            const palette_index = (packed_green >> @intCast(slot * bits_per_index)) & index_mask;
            if (palette_index >= palette.len) return error.InvalidWebpData;
            pixels[written] = palette[palette_index];
            written += 1;
        }
    }
    if (written != output_len) return error.InvalidWebpData;

    return .{
        .allocator = allocator,
        .width = output_width,
        .height = output_height,
        .end_bit_pos = end_bit_pos,
        .pixels = pixels,
    };
}
