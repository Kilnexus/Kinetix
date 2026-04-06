const std = @import("std");
const imaging = @import("Pixio");
pub const ImageError = anyerror;

pub const ImageU8 = imaging.ImageU8;

pub fn loadRgb8(allocator: std.mem.Allocator, path: []const u8) !ImageU8 {
    return imaging.decodeFileRgb8(allocator, path);
}
