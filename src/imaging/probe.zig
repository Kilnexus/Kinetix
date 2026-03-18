const std = @import("std");
const format = @import("format.zig");
const png = @import("codecs/png.zig");
const bmp = @import("codecs/bmp.zig");
const jpeg = @import("codecs/jpeg.zig");
const gif = @import("codecs/gif.zig");
const ico = @import("codecs/ico.zig");
const webp = @import("codecs/webp.zig");

pub const ImageFormat = format.ImageFormat;

pub const ImageInfo = struct {
    format: ImageFormat,
    width: usize,
    height: usize,
    channels: usize,
    has_alpha: bool,
};

pub const WebpInfo = webp.WebpInfo;
pub const WebpChunkTag = webp.WebpChunkTag;
pub const Vp8lStreamInfo = webp.Vp8lStreamInfo;
pub const Vp8lTransformType = webp.Vp8lTransformType;
pub const Vp8lImageRole = webp.Vp8lImageRole;
pub const Vp8lImageDataHeader = webp.Vp8lImageDataHeader;
pub const Vp8lPrefixCodeKind = webp.Vp8lPrefixCodeKind;
pub const Vp8lSimplePrefixCode = webp.Vp8lSimplePrefixCode;
pub const Vp8lNormalPrefixCode = webp.Vp8lNormalPrefixCode;
pub const Vp8lCanonicalCodeEntry = webp.Vp8lCanonicalCodeEntry;
pub const Vp8lCanonicalPrefixSummary = webp.Vp8lCanonicalPrefixSummary;
pub const Vp8lCanonicalSymbolStream = webp.Vp8lCanonicalSymbolStream;
pub const Vp8lPrefixCodeGroupDetail = webp.Vp8lPrefixCodeGroupDetail;
pub const Vp8lEventKind = webp.Vp8lEventKind;
pub const Vp8lEvent = webp.Vp8lEvent;
pub const Vp8lEventStream = webp.Vp8lEventStream;
pub const Vp8lArgbImage = webp.Vp8lArgbImage;
pub const Vp8lPrefixCodeHeader = webp.Vp8lPrefixCodeHeader;
pub const Vp8lPrefixCodeGroup = webp.Vp8lPrefixCodeGroup;
pub const Vp8lEntropyImageDataHeader = webp.Vp8lEntropyImageDataHeader;

pub const ProbeError =
    png.PngError ||
    bmp.BmpError ||
    jpeg.JpegError ||
    gif.GifError ||
    ico.IcoError ||
    webp.WebpError ||
    error{
        UnsupportedImageFormat,
    };

pub fn probeInfo(bytes: []const u8) !ImageInfo {
    return switch (format.detectFormat(bytes)) {
        .png => probePng(bytes),
        .bmp => probeBmp(bytes),
        .jpeg => try probeJpeg(bytes),
        .gif => probeGif(bytes),
        .ico => probeIco(bytes),
        .webp => try probeWebp(bytes),
        else => error.UnsupportedImageFormat,
    };
}

pub fn probeFileInfo(allocator: std.mem.Allocator, path: []const u8) !ImageInfo {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024 * 1024);
    defer allocator.free(bytes);
    return probeInfo(bytes);
}

pub fn probeWebpInfo(bytes: []const u8) !WebpInfo {
    if (format.detectFormat(bytes) != .webp) return error.UnsupportedImageFormat;
    return webp.probeInfo(bytes);
}

pub fn probeWebpFileInfo(allocator: std.mem.Allocator, path: []const u8) !WebpInfo {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024 * 1024);
    defer allocator.free(bytes);
    return probeWebpInfo(bytes);
}

pub fn probeWebpPrimaryChunkTag(bytes: []const u8) !WebpChunkTag {
    if (format.detectFormat(bytes) != .webp) return error.UnsupportedImageFormat;
    return (try webp.findPrimaryChunk(bytes)).tag;
}

pub fn inspectWebpVp8l(bytes: []const u8) !Vp8lStreamInfo {
    if (format.detectFormat(bytes) != .webp) return error.UnsupportedImageFormat;
    return webp.inspectVp8l(bytes);
}

pub fn inspectVp8lImageDataAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    width: usize,
    height: usize,
    role: Vp8lImageRole,
) !Vp8lImageDataHeader {
    return webp.inspectVp8lImageDataAtBitPos(payload, start_bit_pos, width, height, role);
}

pub fn inspectVp8lNormalPrefixCodeAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    alphabet_size: usize,
) !Vp8lNormalPrefixCode {
    return webp.inspectVp8lNormalPrefixCodeAtBitPos(payload, start_bit_pos, alphabet_size);
}

pub fn inspectVp8lCanonicalSymbolStreamAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    code_lengths: []const u8,
    symbol_count: usize,
) !Vp8lCanonicalSymbolStream {
    return webp.inspectVp8lCanonicalSymbolStreamAtBitPos(payload, start_bit_pos, code_lengths, symbol_count);
}

pub fn inspectVp8lPrefixCodeGroupAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    alphabet_sizes: [5]usize,
) !Vp8lPrefixCodeGroupDetail {
    return webp.inspectVp8lPrefixCodeGroupAtBitPos(payload, start_bit_pos, alphabet_sizes);
}

pub fn inspectVp8lEventStreamAtBitPos(
    payload: []const u8,
    prefix_group_start_bit_pos: usize,
    alphabet_sizes: [5]usize,
    width: usize,
    height: usize,
    color_cache_bits: usize,
    max_events: usize,
) !Vp8lEventStream {
    return webp.inspectVp8lEventStreamAtBitPos(
        payload,
        prefix_group_start_bit_pos,
        alphabet_sizes,
        width,
        height,
        color_cache_bits,
        max_events,
    );
}

pub fn resolveMetaPrefixCode(
    entropy_image: ?[]const u32,
    prefix_bits: usize,
    prefix_image_width: usize,
    x: usize,
    y: usize,
) !usize {
    return webp.resolveMetaPrefixCode(entropy_image, prefix_bits, prefix_image_width, x, y);
}

pub fn decodeVp8lSingleGroupArgbAtBitPos(
    allocator: std.mem.Allocator,
    payload: []const u8,
    prefix_group_start_bit_pos: usize,
    alphabet_sizes: [5]usize,
    width: usize,
    height: usize,
    color_cache_bits: usize,
) !Vp8lArgbImage {
    return webp.decodeVp8lSingleGroupArgbAtBitPos(
        allocator,
        payload,
        prefix_group_start_bit_pos,
        alphabet_sizes,
        width,
        height,
        color_cache_bits,
    );
}

pub fn decodeVp8lPayloadArgb(allocator: std.mem.Allocator, payload: []const u8) !Vp8lArgbImage {
    return webp.decodeVp8lPayloadArgb(allocator, payload);
}

fn probePng(bytes: []const u8) ImageInfo {
    const width = readU32be(bytes[16..20]);
    const height = readU32be(bytes[20..24]);
    const color_type = bytes[25];
    return .{
        .format = .png,
        .width = width,
        .height = height,
        .channels = 3,
        .has_alpha = color_type == 4 or color_type == 6,
    };
}

fn probeBmp(bytes: []const u8) ImageInfo {
    const width_i = readI32le(bytes[18..22]);
    const height_i = readI32le(bytes[22..26]);
    const bit_count = readU16le(bytes[28..30]);
    return .{
        .format = .bmp,
        .width = @intCast(@abs(width_i)),
        .height = @intCast(@abs(height_i)),
        .channels = 3,
        .has_alpha = bit_count == 32,
    };
}

fn probeGif(bytes: []const u8) ImageInfo {
    return .{
        .format = .gif,
        .width = readU16le(bytes[6..8]),
        .height = readU16le(bytes[8..10]),
        .channels = 3,
        .has_alpha = false,
    };
}

fn probeIco(bytes: []const u8) ImageInfo {
    const count = readU16le(bytes[4..6]);
    var best_width: usize = 0;
    var best_height: usize = 0;
    var best_score: usize = 0;
    var best_alpha = false;

    for (0..count) |i| {
        const offset = 6 + i * 16;
        const width = if (bytes[offset] == 0) @as(usize, 256) else bytes[offset];
        const height = if (bytes[offset + 1] == 0) @as(usize, 256) else bytes[offset + 1];
        const bit_count = readU16le(bytes[offset + 6 .. offset + 8]);
        const score = width * height * @max(bit_count, 1);
        if (score > best_score) {
            best_score = score;
            best_width = width;
            best_height = height;
            best_alpha = bit_count == 32;
        }
    }

    return .{
        .format = .ico,
        .width = best_width,
        .height = best_height,
        .channels = 3,
        .has_alpha = best_alpha,
    };
}

fn probeJpeg(bytes: []const u8) !ImageInfo {
    if (bytes.len < 4 or bytes[0] != 0xff or bytes[1] != 0xd8) return error.InvalidJpegHeader;

    var pos: usize = 2;
    while (pos + 1 < bytes.len) {
        while (pos < bytes.len and bytes[pos] != 0xff) : (pos += 1) {}
        if (pos + 1 >= bytes.len) break;
        while (pos < bytes.len and bytes[pos] == 0xff) : (pos += 1) {}
        if (pos >= bytes.len) break;

        const marker = bytes[pos];
        pos += 1;

        if (marker == 0xd9 or marker == 0xda) break;
        if (marker >= 0xd0 and marker <= 0xd7) continue;
        if (pos + 2 > bytes.len) return error.InvalidJpegData;
        const seg_len = readU16be(bytes[pos .. pos + 2]);
        if (seg_len < 2 or pos + seg_len > bytes.len) return error.InvalidJpegSegment;

        if ((marker >= 0xc0 and marker <= 0xc3) or (marker >= 0xc5 and marker <= 0xc7) or (marker >= 0xc9 and marker <= 0xcb) or (marker >= 0xcd and marker <= 0xcf)) {
            if (seg_len < 8) return error.InvalidJpegSegment;
            const height = readU16be(bytes[pos + 3 .. pos + 5]);
            const width = readU16be(bytes[pos + 5 .. pos + 7]);
            const components = bytes[pos + 7];
            return .{
                .format = .jpeg,
                .width = width,
                .height = height,
                .channels = if (components == 1) 1 else 3,
                .has_alpha = false,
            };
        }

        pos += seg_len;
    }

    return error.MissingJpegFrame;
}

fn probeWebp(bytes: []const u8) !ImageInfo {
    const info = try webp.probeInfo(bytes);
    return .{
        .format = .webp,
        .width = info.width,
        .height = info.height,
        .channels = 3,
        .has_alpha = info.has_alpha,
    };
}

fn readU16le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) | (@as(usize, bytes[1]) << 8);
}

fn readU16be(bytes: []const u8) usize {
    return (@as(usize, bytes[0]) << 8) | @as(usize, bytes[1]);
}

fn readU32be(bytes: []const u8) usize {
    return (@as(usize, bytes[0]) << 24) |
        (@as(usize, bytes[1]) << 16) |
        (@as(usize, bytes[2]) << 8) |
        @as(usize, bytes[3]);
}

fn readI32le(bytes: []const u8) i32 {
    return @bitCast(@as(u32, @intCast(
        @as(u32, bytes[0]) |
            (@as(u32, bytes[1]) << 8) |
            (@as(u32, bytes[2]) << 16) |
            (@as(u32, bytes[3]) << 24),
    )));
}
