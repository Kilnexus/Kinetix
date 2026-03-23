const bitreader = @import("bitreader.zig");
const jpeg_types = @import("types.zig");

const BitReader = bitreader.BitReader;
const HuffmanTable = jpeg_types.HuffmanTable;
const zigzag = jpeg_types.zigzag;

pub fn decodeBlock(
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
