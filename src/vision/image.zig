const std = @import("std");
const imaging = @import("imaging");

const c = @cImport({
    @cInclude("stb_image.h");
});

pub const ImageError = imaging.ImageError || error{
    ImageDecodeFailed,
};

pub const ImageU8 = imaging.ImageU8;

pub fn loadRgb8(allocator: std.mem.Allocator, path: []const u8) !ImageU8 {
    var path_z = try allocator.allocSentinel(u8, path.len, 0);
    defer allocator.free(path_z);
    @memcpy(path_z[0..path.len], path);

    var width: c_int = 0;
    var height: c_int = 0;
    var channels_in_file: c_int = 0;
    const pixels = c.stbi_load(path_z.ptr, &width, &height, &channels_in_file, 3) orelse return error.ImageDecodeFailed;

    if (width <= 0 or height <= 0) {
        c.stbi_image_free(pixels);
        return error.InvalidImageDimensions;
    }

    const width_usize: usize = @intCast(width);
    const height_usize: usize = @intCast(height);
    var image = try ImageU8.init(allocator, width_usize, height_usize, 3);
    errdefer image.deinit();

    @memcpy(image.data, pixels[0 .. width_usize * height_usize * 3]);
    c.stbi_image_free(pixels);
    return image;
}
