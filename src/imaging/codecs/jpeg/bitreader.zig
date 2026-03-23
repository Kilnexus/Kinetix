pub const BitReader = struct {
    bytes: []const u8,
    pos: usize,
    bit_buffer: u32 = 0,
    bit_count: u5 = 0,

    pub fn readBit(self: *BitReader) !u1 {
        if (self.bit_count == 0) {
            self.bit_buffer = try self.readEntropyByte();
            self.bit_count = 8;
        }
        self.bit_count -= 1;
        return @intCast((self.bit_buffer >> self.bit_count) & 1);
    }

    pub fn readBits(self: *BitReader, count: u8) !u32 {
        var value: u32 = 0;
        for (0..count) |_| {
            value = (value << 1) | try self.readBit();
        }
        return value;
    }

    pub fn alignToByte(self: *BitReader) void {
        self.bit_buffer = 0;
        self.bit_count = 0;
    }

    pub fn consumeRestart(self: *BitReader) !void {
        const marker = try self.readMarker();
        if (marker < 0xD0 or marker > 0xD7) return error.InvalidJpegMarker;
    }

    fn readEntropyByte(self: *BitReader) !u8 {
        if (self.pos >= self.bytes.len) return error.InvalidJpegData;
        const value = self.bytes[self.pos];
        self.pos += 1;
        if (value != 0xFF) return value;
        if (self.pos >= self.bytes.len) return error.InvalidJpegData;

        var marker = self.bytes[self.pos];
        self.pos += 1;
        while (marker == 0xFF) {
            if (self.pos >= self.bytes.len) return error.InvalidJpegData;
            marker = self.bytes[self.pos];
            self.pos += 1;
        }

        if (marker == 0x00) return 0xFF;
        return error.InvalidJpegData;
    }

    fn readMarker(self: *BitReader) !u8 {
        while (self.pos < self.bytes.len) : (self.pos += 1) {
            if (self.bytes[self.pos] != 0xFF) continue;
            self.pos += 1;
            while (self.pos < self.bytes.len and self.bytes[self.pos] == 0xFF) : (self.pos += 1) {}
            if (self.pos >= self.bytes.len) return error.InvalidJpegMarker;
            const marker = self.bytes[self.pos];
            self.pos += 1;
            if (marker == 0x00) continue;
            return marker;
        }
        return error.InvalidJpegMarker;
    }
};
