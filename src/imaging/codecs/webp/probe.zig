const container = @import("container.zig");
const types = @import("types.zig");

pub const WebpInfo = types.WebpInfo;
pub const WebpChunk = types.WebpChunk;

pub const WebpScan = struct {
    info: WebpInfo,
    primary: WebpChunk,
};

pub fn probeInfo(bytes: []const u8) !WebpInfo {
    return (try scanChunks(bytes)).info;
}

pub fn findPrimaryChunk(bytes: []const u8) !WebpChunk {
    return (try scanChunks(bytes)).primary;
}

pub fn scanChunks(bytes: []const u8) !WebpScan {
    try container.validateHeader(bytes);
    var it = try container.ChunkIterator.init(bytes);
    var vp8x_info: ?WebpInfo = null;
    var primary_info: ?WebpInfo = null;
    var primary_chunk: ?WebpChunk = null;

    while (try it.next()) |chunk| {
        switch (chunk.tag) {
            .vp8x => vp8x_info = try parseVp8x(chunk.payload),
            .vp8 => {
                primary_info = try parseVp8(chunk.payload);
                primary_chunk = chunk;
            },
            .vp8l => {
                primary_info = try parseVp8l(chunk.payload);
                primary_chunk = chunk;
            },
            else => {},
        }
    }

    if (primary_info) |info| {
        var resolved = info;
        if (vp8x_info) |extended| {
            resolved.has_alpha = extended.has_alpha;
            resolved.is_animated = extended.is_animated;
            resolved.has_icc = extended.has_icc;
            resolved.has_exif = extended.has_exif;
            resolved.has_xmp = extended.has_xmp;
            resolved.width = extended.width;
            resolved.height = extended.height;
        }
        return .{
            .info = resolved,
            .primary = primary_chunk.?,
        };
    }

    return error.MissingWebpChunk;
}

pub fn parseVp8(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;
    if (payload[0] & 0x01 != 0) return error.UnsupportedWebpBitstream;
    if (!@import("std").mem.eql(u8, payload[3..6], "\x9d\x01\x2a")) return error.InvalidWebpData;

    const width = container.readU16le(payload[6..8]) & 0x3fff;
    const height = container.readU16le(payload[8..10]) & 0x3fff;
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

pub fn parseVp8l(payload: []const u8) !WebpInfo {
    if (payload.len < 5) return error.InvalidWebpData;
    if (payload[0] != 0x2f) return error.InvalidWebpData;

    const bits = container.readU32le(payload[1..5]);
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

pub fn parseVp8x(payload: []const u8) !WebpInfo {
    if (payload.len < 10) return error.InvalidWebpData;

    return .{
        .width = 1 + container.readU24le(payload[4..7]),
        .height = 1 + container.readU24le(payload[7..10]),
        .has_alpha = (payload[0] & 0x10) != 0,
        .is_animated = (payload[0] & 0x02) != 0,
        .has_icc = (payload[0] & 0x20) != 0,
        .has_exif = (payload[0] & 0x08) != 0,
        .has_xmp = (payload[0] & 0x04) != 0,
        .kind = .vp8x,
    };
}
