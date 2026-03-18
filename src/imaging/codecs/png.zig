const std = @import("std");
const types = @import("../types.zig");

pub const ImageU8 = types.ImageU8;

pub const PngError = types.ImageError || error{
    InvalidPngSignature,
    InvalidPngChunk,
    UnsupportedPngBitDepth,
    UnsupportedPngColorType,
    UnsupportedPngInterlace,
    MissingPngIhdr,
    MissingPngIdat,
    MissingPngIend,
    InvalidPngCrc,
    InvalidPngFilter,
    InvalidPngDimensions,
    InvalidPngData,
};

const PngSignature = "\x89PNG\r\n\x1a\n";

pub fn decodeRgb8(allocator: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    if (bytes.len < PngSignature.len or !std.mem.eql(u8, bytes[0..PngSignature.len], PngSignature)) {
        return error.InvalidPngSignature;
    }

    var width: usize = 0;
    var height: usize = 0;
    var bit_depth: u8 = 0;
    var color_type: u8 = 0;
    var compression_method: u8 = 0;
    var filter_method: u8 = 0;
    var interlace_method: u8 = 0;
    var seen_ihdr = false;
    var seen_iend = false;

    var idat = std.ArrayListUnmanaged(u8).empty;
    defer if (idat.items.len > 0) idat.deinit(allocator);

    var offset: usize = PngSignature.len;
    while (offset + 12 <= bytes.len) {
        const chunk_len = readU32be(bytes[offset..][0..4]);
        offset += 4;
        const chunk_type = bytes[offset..][0..4];
        offset += 4;
        if (offset + chunk_len + 4 > bytes.len) return error.InvalidPngChunk;
        const chunk_data = bytes[offset .. offset + chunk_len];
        offset += chunk_len;
        const chunk_crc = readU32be(bytes[offset..][0..4]);
        offset += 4;

        var crc = std.hash.Crc32.init();
        crc.update(chunk_type);
        crc.update(chunk_data);
        if (crc.final() != chunk_crc) return error.InvalidPngCrc;

        if (std.mem.eql(u8, chunk_type, "IHDR")) {
            if (chunk_data.len != 13) return error.InvalidPngChunk;
            width = readU32be(chunk_data[0..4]);
            height = readU32be(chunk_data[4..8]);
            if (width == 0 or height == 0) return error.InvalidPngDimensions;
            bit_depth = chunk_data[8];
            color_type = chunk_data[9];
            compression_method = chunk_data[10];
            filter_method = chunk_data[11];
            interlace_method = chunk_data[12];
            seen_ihdr = true;
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            try idat.appendSlice(allocator, chunk_data);
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            seen_iend = true;
            break;
        }
    }

    if (!seen_ihdr) return error.MissingPngIhdr;
    if (idat.items.len == 0) return error.MissingPngIdat;
    if (!seen_iend) return error.MissingPngIend;
    if (compression_method != 0 or filter_method != 0) return error.InvalidPngData;
    if (bit_depth != 8) return error.UnsupportedPngBitDepth;
    if (interlace_method != 0) return error.UnsupportedPngInterlace;

    const src_channels = switch (color_type) {
        0 => @as(usize, 1),
        2 => @as(usize, 3),
        6 => @as(usize, 4),
        else => return error.UnsupportedPngColorType,
    };
    const bytes_per_pixel = src_channels;
    const scanline_len = width * src_channels;
    const expected_unfiltered_len = height * (1 + scanline_len);

    var in_reader: std.Io.Reader = .fixed(idat.items);
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var decompress: std.compress.flate.Decompress = .init(&in_reader, .zlib, &.{});
    _ = try decompress.reader.streamRemaining(&out.writer);
    const inflated = out.written();
    if (inflated.len != expected_unfiltered_len) return error.InvalidPngData;

    var raw = try allocator.alloc(u8, height * scanline_len);
    defer allocator.free(raw);

    unfilter(raw, inflated, width, height, scanline_len, bytes_per_pixel) catch return error.InvalidPngFilter;

    var image = try ImageU8.init(allocator, width, height, 3);
    errdefer image.deinit();

    for (0..height) |y| {
        for (0..width) |x| {
            const src_index = y * scanline_len + x * src_channels;
            const dst_index = image.pixelIndex(x, y, 0);
            switch (color_type) {
                0 => {
                    const g = raw[src_index];
                    image.data[dst_index] = g;
                    image.data[dst_index + 1] = g;
                    image.data[dst_index + 2] = g;
                },
                2 => {
                    @memcpy(image.data[dst_index .. dst_index + 3], raw[src_index .. src_index + 3]);
                },
                6 => {
                    @memcpy(image.data[dst_index .. dst_index + 3], raw[src_index .. src_index + 3]);
                },
                else => unreachable,
            }
        }
    }
    return image;
}

fn unfilter(
    dst: []u8,
    inflated: []const u8,
    width: usize,
    height: usize,
    scanline_len: usize,
    bytes_per_pixel: usize,
) !void {
    _ = width;
    var src_offset: usize = 0;
    for (0..height) |row| {
        const filter_type = inflated[src_offset];
        src_offset += 1;
        const filtered = inflated[src_offset .. src_offset + scanline_len];
        src_offset += scanline_len;
        const out_row = dst[row * scanline_len .. (row + 1) * scanline_len];
        const prev_row = if (row == 0) null else dst[(row - 1) * scanline_len .. row * scanline_len];

        switch (filter_type) {
            0 => @memcpy(out_row, filtered),
            1 => for (out_row, filtered, 0..) |*d, f, i| {
                const left: u8 = if (i >= bytes_per_pixel) out_row[i - bytes_per_pixel] else 0;
                d.* = f +% left;
            },
            2 => for (out_row, filtered, 0..) |*d, f, i| {
                const up: u8 = if (prev_row) |p| p[i] else 0;
                d.* = f +% up;
            },
            3 => for (out_row, filtered, 0..) |*d, f, i| {
                const left: u8 = if (i >= bytes_per_pixel) out_row[i - bytes_per_pixel] else 0;
                const up: u8 = if (prev_row) |p| p[i] else 0;
                d.* = f +% @as(u8, @intCast((@as(u16, left) + @as(u16, up)) / 2));
            },
            4 => for (out_row, filtered, 0..) |*d, f, i| {
                const left: u8 = if (i >= bytes_per_pixel) out_row[i - bytes_per_pixel] else 0;
                const up: u8 = if (prev_row) |p| p[i] else 0;
                const up_left: u8 = if (prev_row != null and i >= bytes_per_pixel) prev_row.?[i - bytes_per_pixel] else 0;
                d.* = f +% paeth(left, up, up_left);
            },
            else => return error.InvalidPngFilter,
        }
    }
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p = @as(i32, a) + @as(i32, b) - @as(i32, c);
    const pa = @abs(p - @as(i32, a));
    const pb = @abs(p - @as(i32, b));
    const pc = @abs(p - @as(i32, c));
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

fn readU32be(bytes: []const u8) usize {
    return @intCast(std.mem.readInt(u32, bytes[0..4], .big));
}
