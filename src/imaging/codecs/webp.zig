const std = @import("std");
const types = @import("../types.zig");

pub const ImageU8 = types.ImageU8;

pub const WebpKind = enum {
    vp8,
    vp8l,
    vp8x,
};

pub const WebpChunkTag = enum {
    vp8,
    vp8l,
    vp8x,
    alph,
    anim,
    anmf,
    iccp,
    exif,
    xmp,
    unknown,
};

pub const WebpChunk = struct {
    tag: WebpChunkTag,
    payload: []const u8,
};

pub const WebpInfo = struct {
    width: usize,
    height: usize,
    has_alpha: bool,
    is_animated: bool,
    has_icc: bool,
    has_exif: bool,
    has_xmp: bool,
    kind: WebpKind,
};

pub const WebpError = types.ImageError || error{
    InvalidWebpHeader,
    InvalidWebpChunk,
    InvalidWebpData,
    MissingWebpChunk,
    UnsupportedWebpAnimation,
    UnsupportedWebpBitstream,
};

pub fn decodeRgb8(_: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    const scan = try scanChunks(bytes);
    if (scan.info.is_animated) return error.UnsupportedWebpAnimation;
    return switch (scan.primary.tag) {
        .vp8, .vp8l, .vp8x => error.UnsupportedWebpBitstream,
        else => error.MissingWebpChunk,
    };
}

pub fn probeInfo(bytes: []const u8) !WebpInfo {
    return (try scanChunks(bytes)).info;
}

pub fn findPrimaryChunk(bytes: []const u8) !WebpChunk {
    return (try scanChunks(bytes)).primary;
}

pub const ChunkIterator = struct {
    bytes: []const u8,
    pos: usize = 12,

    pub fn init(bytes: []const u8) !ChunkIterator {
        try validateHeader(bytes);
        return .{ .bytes = bytes };
    }

    pub fn next(self: *ChunkIterator) !?WebpChunk {
        if (self.pos + 8 > self.bytes.len) return null;

        const raw_tag = self.bytes[self.pos .. self.pos + 4];
        const chunk_size = readU32le(self.bytes[self.pos + 4 .. self.pos + 8]);
        const payload_offset = self.pos + 8;
        const payload_end = payload_offset + chunk_size;
        if (payload_end > self.bytes.len) return error.InvalidWebpChunk;

        const chunk = WebpChunk{
            .tag = mapChunkTag(raw_tag),
            .payload = self.bytes[payload_offset..payload_end],
        };
        self.pos = payload_end + (chunk_size & 1);
        return chunk;
    }
};

const WebpScan = struct {
    info: WebpInfo,
    primary: WebpChunk,
};

fn scanChunks(bytes: []const u8) !WebpScan {
    try validateHeader(bytes);
    var it = try ChunkIterator.init(bytes);
    var vp8x_info: ?WebpInfo = null;
    var primary_info: ?WebpInfo = null;
    var primary_chunk: ?WebpChunk = null;

    while (try it.next()) |chunk| {
        switch (chunk.tag) {
            .vp8x => vp8x_info = try parseVp8x(chunk.payload),
            .vp8 => {
                primary_info = try parseVp8(chunk.payload);
                primary_chunk = chunk;
                break;
            },
            .vp8l => {
                primary_info = try parseVp8l(chunk.payload);
                primary_chunk = chunk;
                break;
            },
            else => {},
        }
    }

    if (primary_info) |info| {
        if (vp8x_info) |extended| {
            return .{
                .info = .{
                    .width = extended.width,
                    .height = extended.height,
                    .has_alpha = extended.has_alpha or info.has_alpha,
                    .is_animated = extended.is_animated,
                    .has_icc = extended.has_icc,
                    .has_exif = extended.has_exif,
                    .has_xmp = extended.has_xmp,
                    .kind = info.kind,
                },
                .primary = primary_chunk.?,
            };
        }
        return .{
            .info = info,
            .primary = primary_chunk.?,
        };
    }

    if (vp8x_info) |info| {
        return .{
            .info = info,
            .primary = .{
                .tag = .vp8x,
                .payload = &.{},
            },
        };
    }
    return error.MissingWebpChunk;
}

pub fn validateHeader(bytes: []const u8) !void {
    if (bytes.len < 12) return error.InvalidWebpHeader;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WEBP")) {
        return error.InvalidWebpHeader;
    }
}

fn parseVp8(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;
    if (payload[0] & 0x01 != 0) return error.UnsupportedWebpBitstream;
    if (!std.mem.eql(u8, payload[3..6], "\x9d\x01\x2a")) return error.InvalidWebpData;

    const width = readU16le(payload[6..8]) & 0x3fff;
    const height = readU16le(payload[8..10]) & 0x3fff;
    if (width == 0 or height == 0) return error.InvalidWebpData;

    return .{
        .width = width,
        .height = height,
        .has_alpha = false,
        .is_animated = false,
        .has_icc = false,
        .has_exif = false,
        .has_xmp = false,
        .kind = .vp8,
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
        .is_animated = false,
        .has_icc = false,
        .has_exif = false,
        .has_xmp = false,
        .kind = .vp8l,
    };
}

fn parseVp8x(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;

    return .{
        .width = 1 + readU24le(payload[4..7]),
        .height = 1 + readU24le(payload[7..10]),
        .has_alpha = (payload[0] & 0x10) != 0,
        .is_animated = (payload[0] & 0x02) != 0,
        .has_icc = (payload[0] & 0x20) != 0,
        .has_exif = (payload[0] & 0x08) != 0,
        .has_xmp = (payload[0] & 0x04) != 0,
        .kind = .vp8x,
    };
}

fn mapChunkTag(raw: []const u8) WebpChunkTag {
    if (std.mem.eql(u8, raw, "VP8 ")) return .vp8;
    if (std.mem.eql(u8, raw, "VP8L")) return .vp8l;
    if (std.mem.eql(u8, raw, "VP8X")) return .vp8x;
    if (std.mem.eql(u8, raw, "ALPH")) return .alph;
    if (std.mem.eql(u8, raw, "ANIM")) return .anim;
    if (std.mem.eql(u8, raw, "ANMF")) return .anmf;
    if (std.mem.eql(u8, raw, "ICCP")) return .iccp;
    if (std.mem.eql(u8, raw, "EXIF")) return .exif;
    if (std.mem.eql(u8, raw, "XMP ")) return .xmp;
    return .unknown;
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
