const std = @import("std");
const types = @import("types.zig");

pub const WebpChunk = types.WebpChunk;
pub const WebpChunkTag = types.WebpChunkTag;

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

pub fn validateHeader(bytes: []const u8) !void {
    if (bytes.len < 12) return error.InvalidWebpHeader;
    if (!std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WEBP")) {
        return error.InvalidWebpHeader;
    }
}

pub fn mapChunkTag(raw: []const u8) WebpChunkTag {
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

pub fn readU24le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) | (@as(usize, bytes[1]) << 8) | (@as(usize, bytes[2]) << 16);
}

pub fn readU16le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) | (@as(usize, bytes[1]) << 8);
}

pub fn readU32le(bytes: []const u8) usize {
    return @as(usize, bytes[0]) |
        (@as(usize, bytes[1]) << 8) |
        (@as(usize, bytes[2]) << 16) |
        (@as(usize, bytes[3]) << 24);
}
