const std = @import("std");
const types = @import("../types.zig");
const png = @import("png.zig");

pub const ImageU8 = types.ImageU8;

pub const IcoError = types.ImageError || png.PngError || error{
    InvalidIcoHeader,
    InvalidIcoDirectory,
    InvalidIcoPayload,
    MissingIcoImage,
    UnsupportedIcoPayload,
};

const IconDirEntry = struct {
    width: usize,
    height: usize,
    color_count: u8,
    planes: u16,
    bit_count: u16,
    bytes_in_res: usize,
    image_offset: usize,

    fn score(self: IconDirEntry) usize {
        return self.width * self.height * @max(@as(usize, self.bit_count), 1);
    }
};

pub fn decodeRgb8(allocator: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    if (bytes.len < 6) return error.InvalidIcoHeader;
    const reserved = readU16le(bytes[0..2]);
    const image_type = readU16le(bytes[2..4]);
    const count = readU16le(bytes[4..6]);
    if (reserved != 0 or image_type != 1 or count == 0) return error.InvalidIcoHeader;
    if (bytes.len < 6 + @as(usize, count) * 16) return error.InvalidIcoDirectory;

    var best: ?IconDirEntry = null;
    for (0..count) |i| {
        const offset = 6 + i * 16;
        const width = iconDim(bytes[offset]);
        const height = iconDim(bytes[offset + 1]);
        const color_count = bytes[offset + 2];
        const planes = readU16le(bytes[offset + 4 .. offset + 6]);
        const bit_count = readU16le(bytes[offset + 6 .. offset + 8]);
        const bytes_in_res = readU32le(bytes[offset + 8 .. offset + 12]);
        const image_offset = readU32le(bytes[offset + 12 .. offset + 16]);

        if (image_offset + bytes_in_res > bytes.len) return error.InvalidIcoDirectory;

        const entry = IconDirEntry{
            .width = width,
            .height = height,
            .color_count = color_count,
            .planes = planes,
            .bit_count = bit_count,
            .bytes_in_res = bytes_in_res,
            .image_offset = image_offset,
        };

        if (best == null or entry.score() > best.?.score()) best = entry;
    }

    if (best == null) return error.MissingIcoImage;
    const entry = best.?;
    _ = entry.color_count;
    _ = entry.planes;

    const payload = bytes[entry.image_offset .. entry.image_offset + entry.bytes_in_res];
    if (payload.len >= 8 and std.mem.eql(u8, payload[0..8], "\x89PNG\r\n\x1a\n")) {
        return png.decodeRgb8(allocator, payload);
    }

    return error.UnsupportedIcoPayload;
}

fn iconDim(raw: u8) usize {
    return if (raw == 0) 256 else raw;
}

fn readU16le(bytes: []const u8) u16 {
    return std.mem.readInt(u16, bytes[0..2], .little);
}

fn readU32le(bytes: []const u8) usize {
    return @intCast(std.mem.readInt(u32, bytes[0..4], .little));
}
