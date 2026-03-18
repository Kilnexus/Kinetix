const std = @import("std");
const types = @import("../types.zig");

pub const ImageU8 = types.ImageU8;

pub const JpegError = types.ImageError || error{
    InvalidJpegHeader,
    InvalidJpegMarker,
    InvalidJpegSegment,
    InvalidJpegDimensions,
    InvalidJpegData,
    MissingJpegFrame,
    MissingJpegScan,
    UnsupportedJpegFrame,
    UnsupportedJpegPrecision,
    UnsupportedJpegComponents,
    UnsupportedJpegQuantization,
    UnsupportedJpegHuffmanTable,
    UnsupportedJpegScan,
    UnsupportedJpegSampling,
};

const zigzag = [64]u8{
    0,  1,  5,  6,  14, 15, 27, 28,
    2,  4,  7,  13, 16, 26, 29, 42,
    3,  8,  12, 17, 25, 30, 41, 43,
    9,  11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
};

const QuantTable = struct {
    defined: bool = false,
    values: [64]u16 = [_]u16{0} ** 64,
};

const HuffmanTable = struct {
    defined: bool = false,
    counts: [16]u8 = [_]u8{0} ** 16,
    symbols: [256]u8 = [_]u8{0} ** 256,
    symbol_count: usize = 0,
    min_code: [17]i32 = [_]i32{-1} ** 17,
    max_code: [17]i32 = [_]i32{-1} ** 17,
    val_ptr: [17]usize = [_]usize{0} ** 17,

    fn build(self: *HuffmanTable) void {
        var code: i32 = 0;
        var next_index: usize = 0;
        for (1..17) |len| {
            self.val_ptr[len] = next_index;
            const count = self.counts[len - 1];
            if (count == 0) {
                self.min_code[len] = -1;
                self.max_code[len] = -1;
            } else {
                self.min_code[len] = code;
                self.max_code[len] = code + @as(i32, count) - 1;
                code += count;
                next_index += count;
            }
            code <<= 1;
        }
    }
};

const FrameComponent = struct {
    id: u8,
    h: u8,
    v: u8,
    quant_table: u8,
    dc_table: u8 = 0,
    ac_table: u8 = 0,
};

const Frame = struct {
    width: usize = 0,
    height: usize = 0,
    components: [3]FrameComponent = undefined,
    component_count: usize = 0,
    max_h: u8 = 0,
    max_v: u8 = 0,
};

const ScanComponent = struct {
    id: u8,
    dc_table: u8,
    ac_table: u8,
};

const ComponentPlane = struct {
    samples: []u8,
    plane_width: usize,
    plane_height: usize,
    actual_width: usize,
    actual_height: usize,
    h: u8,
    v: u8,
    dc_pred: i32 = 0,
};

const Decoder = struct {
    allocator: std.mem.Allocator,
    bytes: []const u8,
    pos: usize = 0,
    frame: Frame = .{},
    quant_tables: [4]QuantTable = [_]QuantTable{.{}} ** 4,
    dc_tables: [4]HuffmanTable = [_]HuffmanTable{.{}} ** 4,
    ac_tables: [4]HuffmanTable = [_]HuffmanTable{.{}} ** 4,
    restart_interval: usize = 0,
    seen_scan: bool = false,

    fn decode(self: *Decoder) !ImageU8 {
        try self.expectMarker(0xD8);
        while (self.pos < self.bytes.len) {
            const marker = try self.nextMarker();
            switch (marker) {
                0xD9 => break,
                0xC0 => try self.parseSof0(),
                0xC2 => return error.UnsupportedJpegFrame,
                0xC4 => try self.parseDht(),
                0xDB => try self.parseDqt(),
                0xDD => try self.parseDri(),
                0xDA => {
                    self.seen_scan = true;
                    return self.parseScan();
                },
                0xE0...0xEF, 0xFE => try self.skipSegment(),
                else => {
                    if (marker >= 0xD0 and marker <= 0xD7) return error.InvalidJpegMarker;
                    try self.skipSegment();
                },
            }
        }
        return error.MissingJpegScan;
    }

    fn expectMarker(self: *Decoder, marker: u8) !void {
        const found = try self.nextMarker();
        if (found != marker) return error.InvalidJpegHeader;
    }

    fn nextMarker(self: *Decoder) !u8 {
        while (self.pos < self.bytes.len and self.bytes[self.pos] != 0xFF) : (self.pos += 1) {}
        if (self.pos >= self.bytes.len) return error.InvalidJpegMarker;
        while (self.pos < self.bytes.len and self.bytes[self.pos] == 0xFF) : (self.pos += 1) {}
        if (self.pos >= self.bytes.len) return error.InvalidJpegMarker;
        const marker = self.bytes[self.pos];
        self.pos += 1;
        return marker;
    }

    fn skipSegment(self: *Decoder) !void {
        const len = try self.readSegmentLength();
        if (self.pos + len > self.bytes.len) return error.InvalidJpegSegment;
        self.pos += len;
    }

    fn parseDqt(self: *Decoder) !void {
        var remaining = try self.readSegmentLength();
        while (remaining > 0) {
            if (remaining < 65) return error.InvalidJpegSegment;
            const info = try self.readByte();
            remaining -= 1;
            const precision = info >> 4;
            const table_id = info & 0x0f;
            if (precision != 0 or table_id >= self.quant_tables.len) return error.UnsupportedJpegQuantization;
            if (remaining < 64) return error.InvalidJpegSegment;
            var table = &self.quant_tables[table_id];
            for (0..64) |i| {
                table.values[zigzag[i]] = try self.readByte();
            }
            table.defined = true;
            remaining -= 64;
        }
    }

    fn parseDht(self: *Decoder) !void {
        var remaining = try self.readSegmentLength();
        while (remaining > 0) {
            if (remaining < 17) return error.InvalidJpegSegment;
            const info = try self.readByte();
            remaining -= 1;
            const class = info >> 4;
            const table_id = info & 0x0f;
            if (class > 1 or table_id >= 4) return error.UnsupportedJpegHuffmanTable;

            var counts = [_]u8{0} ** 16;
            var total: usize = 0;
            for (0..16) |i| {
                counts[i] = try self.readByte();
                total += counts[i];
            }
            remaining -= 16;
            if (remaining < total) return error.InvalidJpegSegment;

            var table = if (class == 0) &self.dc_tables[table_id] else &self.ac_tables[table_id];
            table.* = .{};
            table.counts = counts;
            table.symbol_count = total;
            for (0..total) |i| {
                table.symbols[i] = try self.readByte();
            }
            table.build();
            table.defined = true;
            remaining -= total;
        }
    }

    fn parseDri(self: *Decoder) !void {
        const len = try self.readSegmentLength();
        if (len != 2) return error.InvalidJpegSegment;
        self.restart_interval = try self.readU16be();
    }

    fn parseSof0(self: *Decoder) !void {
        const len = try self.readSegmentLength();
        if (len < 6) return error.InvalidJpegSegment;
        const precision = try self.readByte();
        if (precision != 8) return error.UnsupportedJpegPrecision;
        const height = try self.readU16be();
        const width = try self.readU16be();
        const component_count = try self.readByte();
        if (width == 0 or height == 0) return error.InvalidJpegDimensions;
        if (component_count != 1 and component_count != 3) return error.UnsupportedJpegComponents;
        if (len != 6 + component_count * 3) return error.InvalidJpegSegment;

        var frame = Frame{
            .width = width,
            .height = height,
            .component_count = component_count,
        };
        for (0..component_count) |i| {
            const id = try self.readByte();
            const hv = try self.readByte();
            const quant_table = try self.readByte();
            const h = hv >> 4;
            const v = hv & 0x0f;
            if (h == 0 or v == 0 or quant_table >= 4) return error.UnsupportedJpegSampling;
            frame.components[i] = .{
                .id = id,
                .h = h,
                .v = v,
                .quant_table = quant_table,
            };
            frame.max_h = @max(frame.max_h, h);
            frame.max_v = @max(frame.max_v, v);
        }
        self.frame = frame;
    }

    fn parseScan(self: *Decoder) !ImageU8 {
        if (self.frame.width == 0 or self.frame.height == 0) return error.MissingJpegFrame;

        const len = try self.readSegmentLength();
        if (len < 6) return error.InvalidJpegSegment;
        const scan_component_count = try self.readByte();
        if (scan_component_count != self.frame.component_count) return error.UnsupportedJpegScan;
        if (len != 4 + scan_component_count * 2) return error.InvalidJpegSegment;

        var scan_components: [3]ScanComponent = undefined;
        for (0..scan_component_count) |i| {
            const id = try self.readByte();
            const selectors = try self.readByte();
            scan_components[i] = .{
                .id = id,
                .dc_table = selectors >> 4,
                .ac_table = selectors & 0x0f,
            };
        }
        const spectral_start = try self.readByte();
        const spectral_end = try self.readByte();
        const successive = try self.readByte();
        if (spectral_start != 0 or spectral_end != 63 or successive != 0) return error.UnsupportedJpegScan;

        for (scan_components[0..scan_component_count]) |scan_component| {
            const frame_component = self.findFrameComponent(scan_component.id) orelse return error.UnsupportedJpegScan;
            frame_component.dc_table = scan_component.dc_table;
            frame_component.ac_table = scan_component.ac_table;
            if (frame_component.dc_table >= 4 or frame_component.ac_table >= 4) return error.UnsupportedJpegHuffmanTable;
        }

        return self.decodeEntropy();
    }

    fn decodeEntropy(self: *Decoder) !ImageU8 {
        if (self.frame.max_h == 0 or self.frame.max_v == 0) return error.MissingJpegFrame;

        var planes: [3]ComponentPlane = undefined;
        const mcu_width = @as(usize, self.frame.max_h) * 8;
        const mcu_height = @as(usize, self.frame.max_v) * 8;
        const mcus_x = divCeil(self.frame.width, mcu_width);
        const mcus_y = divCeil(self.frame.height, mcu_height);

        for (0..self.frame.component_count) |i| {
            const component = self.frame.components[i];
            if (!self.quant_tables[component.quant_table].defined) return error.InvalidJpegData;
            if (!self.dc_tables[component.dc_table].defined or !self.ac_tables[component.ac_table].defined) {
                return error.InvalidJpegData;
            }
            const plane_width = mcus_x * @as(usize, component.h) * 8;
            const plane_height = mcus_y * @as(usize, component.v) * 8;
            const actual_width = divCeil(self.frame.width * @as(usize, component.h), @as(usize, self.frame.max_h));
            const actual_height = divCeil(self.frame.height * @as(usize, component.v), @as(usize, self.frame.max_v));
            planes[i] = .{
                .samples = try self.allocator.alloc(u8, plane_width * plane_height),
                .plane_width = plane_width,
                .plane_height = plane_height,
                .actual_width = actual_width,
                .actual_height = actual_height,
                .h = component.h,
                .v = component.v,
            };
            @memset(planes[i].samples, 0);
        }
        defer for (planes[0..self.frame.component_count]) |plane| self.allocator.free(plane.samples);

        var reader = BitReader{
            .bytes = self.bytes,
            .pos = self.pos,
        };
        var restart_countdown = self.restart_interval;

        for (0..mcus_y) |mcu_y| {
            for (0..mcus_x) |mcu_x| {
                for (0..self.frame.component_count) |i| {
                    const component = self.frame.components[i];
                    for (0..component.v) |block_y| {
                        for (0..component.h) |block_x| {
                            var coeffs = [_]i32{0} ** 64;
                            try decodeBlock(
                                &reader,
                                &self.dc_tables[component.dc_table],
                                &self.ac_tables[component.ac_table],
                                &planes[i].dc_pred,
                                &coeffs,
                            );
                            const samples = idctBlock(&coeffs, &self.quant_tables[component.quant_table].values);
                            try writeBlock(
                                &planes[i],
                                mcu_x * @as(usize, component.h) + block_x,
                                mcu_y * @as(usize, component.v) + block_y,
                                &samples,
                            );
                        }
                    }
                }

                if (self.restart_interval > 0) {
                    restart_countdown -= 1;
                    if (restart_countdown == 0) {
                        reader.alignToByte();
                        try reader.consumeRestart();
                        for (0..self.frame.component_count) |i| {
                            planes[i].dc_pred = 0;
                        }
                        restart_countdown = self.restart_interval;
                    }
                }
            }
        }

        self.pos = reader.pos;

        var image = try ImageU8.init(self.allocator, self.frame.width, self.frame.height, 3);
        errdefer image.deinit();

        if (self.frame.component_count == 1) {
            const plane = planes[0];
            for (0..self.frame.height) |y| {
                for (0..self.frame.width) |x| {
                    const sample = plane.samples[y * plane.plane_width + x];
                    const dst = image.pixelIndex(x, y, 0);
                    image.data[dst] = sample;
                    image.data[dst + 1] = sample;
                    image.data[dst + 2] = sample;
                }
            }
            return image;
        }

        const y_plane = planes[0];
        const cb_plane = planes[1];
        const cr_plane = planes[2];
        for (0..self.frame.height) |y| {
            for (0..self.frame.width) |x| {
                const yv = samplePlane(&y_plane, x, y, self.frame.max_h, self.frame.max_v);
                const cbv = samplePlane(&cb_plane, x, y, self.frame.max_h, self.frame.max_v);
                const crv = samplePlane(&cr_plane, x, y, self.frame.max_h, self.frame.max_v);

                const yf = @as(f32, @floatFromInt(yv));
                const cbf = @as(f32, @floatFromInt(cbv)) - 128.0;
                const crf = @as(f32, @floatFromInt(crv)) - 128.0;

                const r = clampToU8(yf + 1.402 * crf);
                const g = clampToU8(yf - 0.344136 * cbf - 0.714136 * crf);
                const b = clampToU8(yf + 1.772 * cbf);

                const dst = image.pixelIndex(x, y, 0);
                image.data[dst] = r;
                image.data[dst + 1] = g;
                image.data[dst + 2] = b;
            }
        }
        return image;
    }

    fn findFrameComponent(self: *Decoder, id: u8) ?*FrameComponent {
        for (self.frame.components[0..self.frame.component_count]) |*component| {
            if (component.id == id) return component;
        }
        return null;
    }

    fn readSegmentLength(self: *Decoder) !usize {
        const len = try self.readU16be();
        if (len < 2) return error.InvalidJpegSegment;
        return len - 2;
    }

    fn readByte(self: *Decoder) !u8 {
        if (self.pos >= self.bytes.len) return error.InvalidJpegData;
        const value = self.bytes[self.pos];
        self.pos += 1;
        return value;
    }

    fn readU16be(self: *Decoder) !usize {
        if (self.pos + 2 > self.bytes.len) return error.InvalidJpegData;
        const value = (@as(u16, self.bytes[self.pos]) << 8) | @as(u16, self.bytes[self.pos + 1]);
        self.pos += 2;
        return value;
    }
};

const BitReader = struct {
    bytes: []const u8,
    pos: usize,
    bit_buffer: u32 = 0,
    bit_count: u5 = 0,

    fn readBit(self: *BitReader) !u1 {
        if (self.bit_count == 0) {
            self.bit_buffer = try self.readEntropyByte();
            self.bit_count = 8;
        }
        self.bit_count -= 1;
        return @intCast((self.bit_buffer >> self.bit_count) & 1);
    }

    fn readBits(self: *BitReader, count: u8) !u32 {
        var value: u32 = 0;
        for (0..count) |_| {
            value = (value << 1) | try self.readBit();
        }
        return value;
    }

    fn alignToByte(self: *BitReader) void {
        self.bit_buffer = 0;
        self.bit_count = 0;
    }

    fn consumeRestart(self: *BitReader) !void {
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

pub fn decodeRgb8(allocator: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    var decoder = Decoder{
        .allocator = allocator,
        .bytes = bytes,
    };
    return decoder.decode();
}

fn decodeBlock(
    reader: *BitReader,
    dc_table: *const HuffmanTable,
    ac_table: *const HuffmanTable,
    dc_pred: *i32,
    coeffs: *[64]i32,
) !void {
    const dc_len = try decodeHuffman(reader, dc_table);
    const dc_diff = try receiveAndExtend(reader, dc_len);
    dc_pred.* += dc_diff;
    coeffs[0] = dc_pred.*;

    var index: usize = 1;
    while (index < 64) {
        const symbol = try decodeHuffman(reader, ac_table);
        const run = symbol >> 4;
        const size = symbol & 0x0f;
        if (size == 0) {
            if (run == 0) break;
            if (run == 15) {
                index += 16;
                continue;
            }
            return error.InvalidJpegData;
        }

        index += run;
        if (index >= 64) return error.InvalidJpegData;
        coeffs[zigzag[index]] = try receiveAndExtend(reader, size);
        index += 1;
    }
}

fn decodeHuffman(reader: *BitReader, table: *const HuffmanTable) !u8 {
    var code: i32 = 0;
    for (1..17) |len| {
        code = (code << 1) | try reader.readBit();
        if (table.max_code[len] >= 0 and code <= table.max_code[len]) {
            const index = table.val_ptr[len] + @as(usize, @intCast(code - table.min_code[len]));
            if (index >= table.symbol_count) return error.InvalidJpegData;
            return table.symbols[index];
        }
    }
    return error.InvalidJpegData;
}

fn receiveAndExtend(reader: *BitReader, count: u8) !i32 {
    if (count == 0) return 0;
    const value = try reader.readBits(count);
    const vt: i32 = @as(i32, 1) << @intCast(count - 1);
    var signed: i32 = @intCast(value);
    if (signed < vt) {
        signed += (-@as(i32, 1) << @intCast(count)) + 1;
    }
    return signed;
}

fn idctBlock(coeffs: *const [64]i32, quant: *const [64]u16) [64]u8 {
    var out = [_]u8{0} ** 64;
    const inv_sqrt2: f64 = 0.7071067811865476;

    for (0..8) |y| {
        for (0..8) |x| {
            var sum: f64 = 0.0;
            for (0..8) |v| {
                const cv = if (v == 0) inv_sqrt2 else 1.0;
                const cos_y = std.math.cos((@as(f64, @floatFromInt(2 * y + 1)) * @as(f64, @floatFromInt(v)) * std.math.pi) / 16.0);
                for (0..8) |u| {
                    const cu = if (u == 0) inv_sqrt2 else 1.0;
                    const cos_x = std.math.cos((@as(f64, @floatFromInt(2 * x + 1)) * @as(f64, @floatFromInt(u)) * std.math.pi) / 16.0);
                    const idx = v * 8 + u;
                    const value = @as(f64, @floatFromInt(coeffs[idx])) * @as(f64, @floatFromInt(quant[idx]));
                    sum += cu * cv * value * cos_x * cos_y;
                }
            }
            out[y * 8 + x] = clampToU8(@as(f32, @floatCast(sum / 4.0 + 128.0)));
        }
    }

    return out;
}

fn writeBlock(plane: *ComponentPlane, block_x: usize, block_y: usize, samples: *const [64]u8) !void {
    const start_x = block_x * 8;
    const start_y = block_y * 8;
    if (start_x + 8 > plane.plane_width or start_y + 8 > plane.plane_height) return error.InvalidJpegData;
    for (0..8) |y| {
        const dst_row = (start_y + y) * plane.plane_width + start_x;
        const src_row = y * 8;
        @memcpy(plane.samples[dst_row .. dst_row + 8], samples[src_row .. src_row + 8]);
    }
}

fn samplePlane(plane: *const ComponentPlane, x: usize, y: usize, max_h: u8, max_v: u8) u8 {
    const sample_x = @min((x * @as(usize, plane.h)) / @as(usize, max_h), plane.actual_width - 1);
    const sample_y = @min((y * @as(usize, plane.v)) / @as(usize, max_v), plane.actual_height - 1);
    return plane.samples[sample_y * plane.plane_width + sample_x];
}

fn divCeil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn clampToU8(value: f32) u8 {
    if (value <= 0) return 0;
    if (value >= 255) return 255;
    return @intFromFloat(@round(value));
}
