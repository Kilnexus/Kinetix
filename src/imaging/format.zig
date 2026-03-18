const std = @import("std");

pub const ImageFormat = enum {
    png,
    bmp,
    jpeg,
    gif,
    unknown,
};

pub fn detectFormat(bytes: []const u8) ImageFormat {
    if (bytes.len >= 8 and std.mem.eql(u8, bytes[0..8], "\x89PNG\r\n\x1a\n")) return .png;
    if (bytes.len >= 2 and bytes[0] == 'B' and bytes[1] == 'M') return .bmp;
    if (bytes.len >= 3 and bytes[0] == 0xff and bytes[1] == 0xd8 and bytes[2] == 0xff) return .jpeg;
    if (bytes.len >= 6 and (std.mem.eql(u8, bytes[0..6], "GIF87a") or std.mem.eql(u8, bytes[0..6], "GIF89a"))) return .gif;
    return .unknown;
}
