const std = @import("std");
const types = @import("../types.zig");
const png = @import("png.zig");

pub const ImageU8 = types.ImageU8;

pub const IcoError = types.ImageError || png.PngError || error{
    InvalidIcoHeader,
    InvalidIcoDirectory,
    InvalidIcoPayload,
    MissingIcoImage,
    UnsupportedIcoBitDepth,
    UnsupportedIcoCompression,
    UnsupportedIcoDibHeader,
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
    if (payload.len >= 40) {
        const dib_size = readU32le(payload[0..4]);
        if (dib_size >= 40 and dib_size <= payload.len) {
            return decodeBmpIconRgb8(allocator, payload);
        }
    }

    return error.UnsupportedIcoPayload;
}

fn decodeBmpIconRgb8(allocator: std.mem.Allocator, payload: []const u8) !ImageU8 {
    if (payload.len < 40) return error.InvalidIcoPayload;

    const dib_size = readU32le(payload[0..4]);
    if (dib_size < 40 or dib_size > payload.len) return error.UnsupportedIcoDibHeader;

    const width_i = readI32le(payload[4..8]);
    const height_i = readI32le(payload[8..12]);
    const planes = readU16le(payload[12..14]);
    const bit_count = readU16le(payload[14..16]);
    const compression = readU32le(payload[16..20]);

    if (planes != 1) return error.InvalidIcoPayload;
    if (compression != 0) return error.UnsupportedIcoCompression;
    if (bit_count != 24 and bit_count != 32) return error.UnsupportedIcoBitDepth;
    if (width_i == 0 or height_i == 0) return error.InvalidIcoPayload;

    const width: usize = @intCast(@abs(width_i));
    const total_height: usize = @intCast(@abs(height_i));
    if (width == 0 or total_height < 2) return error.InvalidIcoPayload;

    const height: usize = if (height_i > 0) total_height / 2 else total_height;
    if (height == 0) return error.InvalidIcoPayload;

    const bottom_up = height_i > 0;
    const xor_channels: usize = if (bit_count == 24) 3 else 4;
    const xor_row_stride = ((width * @as(usize, bit_count) + 31) / 32) * 4;
    const and_row_stride = ((width + 31) / 32) * 4;
    const xor_offset = dib_size;
    const xor_size = xor_row_stride * height;
    if (xor_offset + xor_size > payload.len) return error.InvalidIcoPayload;
    const and_offset = xor_offset + xor_size;
    const and_size = and_row_stride * height;
    if (and_offset + and_size > payload.len) return error.InvalidIcoPayload;

    var image = try ImageU8.init(allocator, width, height, 3);
    errdefer image.deinit();
    image.fill(0);

    const xor_bitmap = payload[xor_offset .. xor_offset + xor_size];
    const and_bitmap = payload[and_offset .. and_offset + and_size];

    for (0..height) |y| {
        const src_y = if (bottom_up) height - 1 - y else y;
        const xor_row = xor_bitmap[src_y * xor_row_stride .. (src_y + 1) * xor_row_stride];
        const and_row = and_bitmap[src_y * and_row_stride .. (src_y + 1) * and_row_stride];

        for (0..width) |x| {
            const src_index = x * xor_channels;
            const mask_byte = and_row[x / 8];
            const mask_bit: u8 = @as(u8, 0x80) >> @as(u3, @intCast(x % 8));
            const masked = (mask_byte & mask_bit) != 0;
            const alpha_zero = bit_count == 32 and xor_row[src_index + 3] == 0;
            if (masked or alpha_zero) continue;

            const dst = image.pixelIndex(x, y, 0);
            image.data[dst] = xor_row[src_index + 2];
            image.data[dst + 1] = xor_row[src_index + 1];
            image.data[dst + 2] = xor_row[src_index];
        }
    }

    return image;
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

fn readI32le(bytes: []const u8) i32 {
    return std.mem.readInt(i32, bytes[0..4], .little);
}
