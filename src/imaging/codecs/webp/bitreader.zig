const types = @import("types.zig");

pub const Vp8lBitReader = struct {
    bytes: []const u8,
    bit_pos: usize = 0,

    pub fn init(bytes: []const u8) Vp8lBitReader {
        return .{ .bytes = bytes };
    }

    pub fn initAtBit(bytes: []const u8, bit_pos: usize) Vp8lBitReader {
        return .{
            .bytes = bytes,
            .bit_pos = bit_pos,
        };
    }

    pub fn readBits(self: *Vp8lBitReader, count: usize) types.WebpError!usize {
        if (count > 24) return error.InvalidWebpData;

        var value: usize = 0;
        for (0..count) |i| {
            const byte_index = self.bit_pos / 8;
            if (byte_index >= self.bytes.len) return error.InvalidWebpData;
            const bit_index: u3 = @intCast(self.bit_pos % 8);
            const bit = (self.bytes[byte_index] >> bit_index) & 1;
            value |= @as(usize, bit) << @intCast(i);
            self.bit_pos += 1;
        }
        return value;
    }
};
