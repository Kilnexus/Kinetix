const std = @import("std");

const c = @cImport({
    @cInclude("stb_image.h");
});

pub const ImageError = error{
    ImageDecodeFailed,
    InvalidImageDimensions,
};

pub const ImageU8 = struct {
    width: usize,
    height: usize,
    channels: usize,
    data: []u8,

    pub fn deinit(self: *ImageU8) void {
        c.stbi_image_free(self.data.ptr);
        self.* = undefined;
    }
};

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
    const len = width_usize * height_usize * 3;

    return .{
        .width = width_usize,
        .height = height_usize,
        .channels = 3,
        .data = pixels[0..len],
    };
}
