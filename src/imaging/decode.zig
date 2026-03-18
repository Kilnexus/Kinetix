const std = @import("std");
const types = @import("types.zig");
const format = @import("format.zig");
const png = @import("codecs/png.zig");
const bmp = @import("codecs/bmp.zig");
const jpeg = @import("codecs/jpeg.zig");
const gif = @import("codecs/gif.zig");
const ico = @import("codecs/ico.zig");
const webp = @import("codecs/webp.zig");

pub const ImageU8 = types.ImageU8;
pub const ImageFormat = format.ImageFormat;

pub const DecodeError = types.ImageError || png.PngError || bmp.BmpError || jpeg.JpegError || gif.GifError || ico.IcoError || webp.WebpError || error{
    UnsupportedImageFormat,
};

pub fn decodeRgb8(allocator: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    return switch (format.detectFormat(bytes)) {
        .png => png.decodeRgb8(allocator, bytes),
        .bmp => bmp.decodeRgb8(allocator, bytes),
        .jpeg => jpeg.decodeRgb8(allocator, bytes),
        .gif => gif.decodeRgb8(allocator, bytes),
        .ico => ico.decodeRgb8(allocator, bytes),
        .webp => webp.decodeRgb8(allocator, bytes),
        else => error.UnsupportedImageFormat,
    };
}

pub fn decodeFileRgb8(allocator: std.mem.Allocator, path: []const u8) !ImageU8 {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024 * 1024);
    defer allocator.free(bytes);
    return decodeRgb8(allocator, bytes);
}
