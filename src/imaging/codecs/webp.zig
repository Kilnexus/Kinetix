const std = @import("std");
const types = @import("../types.zig");

pub const ImageU8 = types.ImageU8;

pub const WebpInfo = struct {
    width: usize,
    height: usize,
    has_alpha: bool,
};

pub const WebpError = types.ImageError || error{
    InvalidWebpHeader,
    InvalidWebpChunk,
    InvalidWebpData,
    MissingWebpChunk,
    UnsupportedWebpBitstream,
};

pub fn decodeRgb8(_: std.mem.Allocator, _: []const u8) !ImageU8 {
    return error.UnsupportedWebpBitstream;
}

pub fn probeInfo(bytes: []const u8) !WebpInfo {
    if (bytes.len < 12) return error.InvalidWebpHeader;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WEBP")) {
        return error.InvalidWebpHeader;
    }

    var pos: usize = 12;
    while (pos + 8 <= bytes.len) {
        const tag = bytes[pos .. pos + 4];
        const chunk_size = readU32le(bytes[pos + 4 .. pos + 8]);
        const payload_offset = pos + 8;
        const payload_end = payload_offset + chunk_size;
        if (payload_end > bytes.len) return error.InvalidWebpChunk;
        const payload = bytes[payload_offset..payload_end];

        if (std.mem.eql(u8, tag, "VP8 ")) return parseVp8(payload);
        if (std.mem.eql(u8, tag, "VP8L")) return parseVp8l(payload);
        if (std.mem.eql(u8, tag, "VP8X")) return parseVp8x(payload);

        pos = payload_end + (chunk_size & 1);
    }

    return error.MissingWebpChunk;
}

fn parseVp8(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;
    if (payload[3] & 0x01 != 0) return error.UnsupportedWebpBitstream;
    if (!std.mem.eql(u8, payload[3..6], "\x9d\x01\x2a")) return error.InvalidWebpData;

    const width = readU16le(payload[6..8]) & 0x3fff;
    const height = readU16le(payload[8..10]) & 0x3fff;
    if (width == 0 or height == 0) return error.InvalidWebpData;

    return .{
        .width = width,
        .height = height,
        .has_alpha = false,
    };
}

fn parseVp8l(payload: []const u8) !WebpInfo {
    if (payload.len < 5) return error.InvalidWebpData;
    if (payload[0] != 0x2f) return error.InvalidWebpData;

    const bits = readU32le(payload[1..5]);
    const width = 1 + (bits & 0x3fff);
    const height = 1 + ((bits >> 14) & 0x3fff);
    const has_alpha = ((bits >> 28) & 0x1) != 0;
    const version = (bits >> 29) & 0x7;
    if (version != 0) return error.UnsupportedWebpBitstream;

    return .{
        .width = width,
        .height = height,
        .has_alpha = has_alpha,
    };
}

fn parseVp8x(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;

    return .{
        .width = 1 + readU24le(payload[4..7]),
        .height = 1 + readU24le(payload[7..10]),
        .has_alpha = (payload[0] & 0x10) != 0,
    };
}

fn readU24le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) | (@as(usize, bytes[1]) << 8) | (@as(usize, bytes[2]) << 16);
}

fn readU16le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) | (@as(usize, bytes[1]) << 8);
}

fn readU32le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) |
        (@as(usize, bytes[1]) << 8) |
        (@as(usize, bytes[2]) << 16) |
        (@as(usize, bytes[3]) << 24);
}
