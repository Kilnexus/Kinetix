const std = @import("std");
const types = @import("types.zig");
const bitreader_mod = @import("bitreader.zig");

pub const Vp8lBitReader = bitreader_mod.Vp8lBitReader;
pub const Vp8lNormalPrefixCode = types.Vp8lNormalPrefixCode;
pub const Vp8lCanonicalPrefixSummary = types.Vp8lCanonicalPrefixSummary;
pub const Vp8lCanonicalCodeEntry = types.Vp8lCanonicalCodeEntry;
pub const Vp8lCanonicalSymbolStream = types.Vp8lCanonicalSymbolStream;
pub const Vp8lPrefixCodeGroup = types.Vp8lPrefixCodeGroup;
pub const Vp8lPrefixCodeGroupDetail = types.Vp8lPrefixCodeGroupDetail;
pub const Vp8lPrefixCodeHeader = types.Vp8lPrefixCodeHeader;
pub const Vp8lSimplePrefixCode = types.Vp8lSimplePrefixCode;

pub const maxPrefixAlphabetSize = 256 + 24 + (1 << 11);
pub const codeLengthLiteralCount = 16;
pub const codeLengthRepeatCode = 16;
pub const defaultCodeLength: u8 = 8;
pub const codeLengthExtraBits = [3]usize{ 2, 3, 7 };
pub const codeLengthRepeatOffsets = [3]usize{ 3, 3, 11 };
pub const numPrefixCodes = 5;
pub const numLengthCodes = 24;
pub const numDistanceCodes = 40;

pub const codeLengthCodeOrder = [19]usize{
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15,
};

pub const RuntimePrefixCodeGroup = struct {
    codes: [numPrefixCodes]CanonicalPrefixDecoder,
};

pub const CanonicalPrefixDecoder = struct {
    const Entry = struct {
        symbol: usize,
        len: usize,
        code: usize,
    };

    entries: [19]Entry = [_]Entry{.{ .symbol = 0, .len = 0, .code = 0 }} ** 19,
    entry_count: usize = 0,
    max_len: usize = 0,

    pub fn init(code_lengths: []const usize) !CanonicalPrefixDecoder {
        return initImpl(usize, code_lengths);
    }

    pub fn initFromU8(code_lengths: []const u8) !CanonicalPrefixDecoder {
        return initImpl(u8, code_lengths);
    }

    fn initImpl(comptime T: type, code_lengths: []const T) !CanonicalPrefixDecoder {
        var counts = [_]usize{0} ** 16;
        for (code_lengths) |len| {
            const len_usize = @as(usize, len);
            if (len_usize >= counts.len) return error.InvalidWebpData;
            if (len_usize != 0) counts[len_usize] += 1;
        }

        var next_code = [_]usize{0} ** 16;
        var code: usize = 0;
        for (1..counts.len) |len| {
            code = (code + counts[len - 1]) << 1;
            next_code[len] = code;
        }

        var decoder = CanonicalPrefixDecoder{};
        for (code_lengths, 0..) |len, symbol| {
            const len_usize = @as(usize, len);
            if (len_usize == 0) continue;
            const canonical_code = next_code[len_usize];
            next_code[len_usize] += 1;
            decoder.entries[decoder.entry_count] = .{
                .symbol = symbol,
                .len = len_usize,
                .code = reverseBits(canonical_code, len_usize),
            };
            decoder.entry_count += 1;
            decoder.max_len = @max(decoder.max_len, len_usize);
        }

        if (decoder.entry_count == 0) return error.InvalidWebpData;
        return decoder;
    }

    pub fn readSymbol(self: *const CanonicalPrefixDecoder, reader: *Vp8lBitReader) !usize {
        if (self.entry_count == 1) return self.entries[0].symbol;
        var acc: usize = 0;
        for (1..self.max_len + 1) |len| {
            acc |= (try reader.readBits(1)) << @intCast(len - 1);
            for (self.entries[0..self.entry_count]) |entry| {
                if (entry.len == len and entry.code == acc) return entry.symbol;
            }
        }
        return error.InvalidWebpData;
    }
};

pub fn inspectVp8lNormalPrefixCodeAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    alphabet_size: usize,
) !Vp8lNormalPrefixCode {
    if (alphabet_size > maxPrefixAlphabetSize) return error.InvalidWebpData;
    var reader = Vp8lBitReader.initAtBit(payload, start_bit_pos);
    return inspectNormalPrefixCodeDetailed(&reader, alphabet_size);
}

pub fn inspectVp8lCanonicalSymbolStreamAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    code_lengths: []const u8,
    symbol_count: usize,
) !Vp8lCanonicalSymbolStream {
    var reader = Vp8lBitReader.initAtBit(payload, start_bit_pos);
    const decoder = try CanonicalPrefixDecoder.initFromU8(code_lengths);
    var preview = [_]usize{0} ** 32;
    const preview_len = @min(preview.len, symbol_count);
    for (0..symbol_count) |i| {
        const symbol = try decoder.readSymbol(&reader);
        if (i < preview_len) preview[i] = symbol;
    }
    return .{
        .start_bit_pos = start_bit_pos,
        .end_bit_pos = reader.bit_pos,
        .symbol_count = symbol_count,
        .preview_len = preview_len,
        .preview = preview,
    };
}

pub fn inspectVp8lPrefixCodeGroupAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    alphabet_sizes: [5]usize,
) !Vp8lPrefixCodeGroupDetail {
    var reader = Vp8lBitReader.initAtBit(payload, start_bit_pos);
    const group = try inspectPrefixCodeGroupDetailed(&reader, alphabet_sizes);
    return .{
        .start_bit_pos = start_bit_pos,
        .end_bit_pos = reader.bit_pos,
        .alphabet_sizes = alphabet_sizes,
        .group = group,
    };
}

pub fn inspectPrefixCodeGroup(reader: *Vp8lBitReader) !Vp8lPrefixCodeGroup {
    return inspectPrefixCodeGroupImpl(reader, null);
}

pub fn inspectPrefixCodeGroupDetailed(reader: *Vp8lBitReader, alphabet_sizes: [5]usize) !Vp8lPrefixCodeGroup {
    return inspectPrefixCodeGroupImpl(reader, alphabet_sizes);
}

pub fn inspectNormalPrefixCode(reader: *Vp8lBitReader) !Vp8lNormalPrefixCode {
    return inspectNormalPrefixCodeImpl(reader, null);
}

pub fn inspectNormalPrefixCodeDetailed(reader: *Vp8lBitReader, alphabet_size: usize) !Vp8lNormalPrefixCode {
    return inspectNormalPrefixCodeImpl(reader, alphabet_size);
}

pub fn parseRuntimePrefixCodeGroup(reader: *Vp8lBitReader, alphabet_sizes: [numPrefixCodes]usize) !RuntimePrefixCodeGroup {
    var decoders = [_]CanonicalPrefixDecoder{undefined} ** numPrefixCodes;
    for (alphabet_sizes, 0..) |alphabet_size, i| {
        decoders[i] = try parseRuntimePrefixCode(reader, alphabet_size);
    }
    return .{ .codes = decoders };
}

pub fn readPrefixCodedValue(symbol: usize, reader: *Vp8lBitReader) !usize {
    if (symbol < 4) return symbol + 1;
    const extra_bits = (symbol - 2) >> 1;
    const offset = ((2 + (symbol & 1)) << @intCast(extra_bits)) + 1;
    return offset + (try reader.readBits(extra_bits));
}

pub fn planeCodeToDistance(width: usize, dist_code: usize) usize {
    if (dist_code <= 120) return planeCodeToDistanceFast(width, dist_code);
    return dist_code - 120;
}

pub fn buildCanonicalPrefixSummary(code_lengths: []const u8) !Vp8lCanonicalPrefixSummary {
    var counts = [_]usize{0} ** 32;
    var max_len: usize = 0;
    var active: usize = 0;
    for (code_lengths) |len_u8| {
        const len = @as(usize, len_u8);
        if (len >= counts.len) return error.InvalidWebpData;
        if (len == 0) continue;
        counts[len] += 1;
        max_len = @max(max_len, len);
        active += 1;
    }
    if (active == 0) return error.InvalidWebpData;

    var next_code = [_]usize{0} ** 32;
    var code: usize = 0;
    for (1..max_len + 1) |len| {
        code = (code + counts[len - 1]) << 1;
        next_code[len] = code;
    }

    var preview = [_]Vp8lCanonicalCodeEntry{.{ .symbol = 0, .len = 0, .lsb_code = 0 }} ** 16;
    var preview_len: usize = 0;
    for (code_lengths, 0..) |len_u8, symbol| {
        const len = @as(usize, len_u8);
        if (len == 0) continue;
        const canonical_code = next_code[len];
        next_code[len] += 1;
        if (preview_len < preview.len) {
            preview[preview_len] = .{
                .symbol = symbol,
                .len = len,
                .lsb_code = reverseBits(canonical_code, len),
            };
            preview_len += 1;
        }
    }

    return .{
        .active_symbol_count = active,
        .max_code_length = max_len,
        .preview_len = preview_len,
        .preview = preview,
    };
}

fn inspectPrefixCodeGroupImpl(reader: *Vp8lBitReader, alphabet_sizes: ?[5]usize) !Vp8lPrefixCodeGroup {
    var codes = [_]Vp8lPrefixCodeHeader{undefined} ** 5;
    var parsed_count: usize = 0;
    var all_simple = true;

    while (parsed_count < codes.len) {
        const start_bit_pos = reader.bit_pos;
        const is_simple = (try reader.readBits(1)) == 1;
        if (!is_simple) {
            all_simple = false;
            const normal = if (alphabet_sizes) |sizes|
                try inspectNormalPrefixCodeDetailed(reader, sizes[parsed_count])
            else
                try inspectNormalPrefixCode(reader);
            codes[parsed_count] = .{
                .kind = .normal,
                .start_bit_pos = start_bit_pos,
                .normal = normal,
            };
            parsed_count += 1;
            continue;
        }

        const num_symbols = (try reader.readBits(1)) + 1;
        const is_first_8bits = (try reader.readBits(1)) == 1;
        const symbol0 = try reader.readBits(if (is_first_8bits) 8 else 1);
        const symbol1: ?usize = if (num_symbols == 2) try reader.readBits(8) else null;
        const canonical_summary = if (alphabet_sizes) |sizes|
            try buildSimplePrefixSummary(num_symbols, symbol0, symbol1, sizes[parsed_count])
        else
            null;
        codes[parsed_count] = .{
            .kind = .simple,
            .start_bit_pos = start_bit_pos,
            .simple = .{
                .num_symbols = num_symbols,
                .is_first_8bits = is_first_8bits,
                .symbol0 = symbol0,
                .symbol1 = symbol1,
                .canonical_summary = canonical_summary,
                .end_bit_pos = reader.bit_pos,
            },
        };
        parsed_count += 1;
    }

    return .{
        .parsed_count = parsed_count,
        .all_simple = all_simple,
        .codes = codes,
    };
}

fn buildSimplePrefixSummary(
    num_symbols: usize,
    symbol0: usize,
    symbol1: ?usize,
    alphabet_size: usize,
) !Vp8lCanonicalPrefixSummary {
    if (alphabet_size > maxPrefixAlphabetSize) return error.InvalidWebpData;
    if (symbol0 >= alphabet_size) return error.InvalidWebpData;
    var code_lengths = [_]u8{0} ** maxPrefixAlphabetSize;
    code_lengths[symbol0] = 1;
    if (num_symbols == 2) {
        const second = symbol1 orelse return error.InvalidWebpData;
        if (second >= alphabet_size) return error.InvalidWebpData;
        code_lengths[second] = 1;
    }
    return buildCanonicalPrefixSummary(code_lengths[0..alphabet_size]);
}

fn inspectNormalPrefixCodeImpl(reader: *Vp8lBitReader, alphabet_size: ?usize) !Vp8lNormalPrefixCode {
    const num_code_length_codes = (try reader.readBits(4)) + 4;
    var code_length_code_lengths = [_]usize{0} ** 19;

    for (0..num_code_length_codes) |i| {
        const symbol = codeLengthCodeOrder[i];
        code_length_code_lengths[symbol] = try reader.readBits(3);
    }

    const use_explicit_max_symbol = (try reader.readBits(1)) == 1;
    const length_nbits = if (use_explicit_max_symbol) 2 + 2 * (try reader.readBits(3)) else null;
    const max_symbol = if (use_explicit_max_symbol) 2 + (try reader.readBits(length_nbits.?)) else alphabet_size orelse 0;

    var info = Vp8lNormalPrefixCode{
        .num_code_length_codes = num_code_length_codes,
        .code_length_code_lengths = code_length_code_lengths,
        .use_explicit_max_symbol = use_explicit_max_symbol,
        .length_nbits = length_nbits,
        .max_symbol = max_symbol,
        .end_bit_pos = reader.bit_pos,
    };

    if (alphabet_size) |resolved_alphabet_size| {
        if (max_symbol > resolved_alphabet_size) return error.InvalidWebpData;
        var code_lengths = [_]u8{0} ** maxPrefixAlphabetSize;
        const summary = try inspectDecodedCodeLengths(reader, code_length_code_lengths, resolved_alphabet_size, max_symbol, &code_lengths);
        info.decoded_symbol_tokens = summary.decoded_symbol_tokens;
        info.emitted_code_lengths = summary.emitted_code_lengths;
        info.non_zero_code_lengths = summary.non_zero_code_lengths;
        info.preview_len = summary.preview_len;
        info.preview = summary.preview;
        info.canonical_summary = try buildCanonicalPrefixSummary(code_lengths[0..resolved_alphabet_size]);
        info.end_bit_pos = reader.bit_pos;
    }

    return info;
}

fn inspectDecodedCodeLengths(
    reader: *Vp8lBitReader,
    code_length_code_lengths: [19]usize,
    alphabet_size: usize,
    max_symbol: usize,
    code_lengths: *[maxPrefixAlphabetSize]u8,
) !struct {
    decoded_symbol_tokens: usize,
    emitted_code_lengths: usize,
    non_zero_code_lengths: usize,
    preview_len: usize,
    preview: [32]u8,
} {
    const decoder = try CanonicalPrefixDecoder.init(code_length_code_lengths[0..]);
    var emitted: usize = 0;
    var tokens: usize = 0;
    var prev_code_len: u8 = defaultCodeLength;

    while (emitted < alphabet_size and tokens < max_symbol) : (tokens += 1) {
        const symbol = try decoder.readSymbol(reader);
        if (symbol < codeLengthLiteralCount) {
            const code_len: u8 = @intCast(symbol);
            code_lengths[emitted] = code_len;
            emitted += 1;
            if (code_len != 0) prev_code_len = code_len;
            continue;
        }

        const slot = symbol - codeLengthLiteralCount;
        const extra_bits = codeLengthExtraBits[slot];
        const repeat_offset = codeLengthRepeatOffsets[slot];
        const repeat = (try reader.readBits(extra_bits)) + repeat_offset;
        if (emitted + repeat > alphabet_size) return error.InvalidWebpData;

        const use_prev = symbol == codeLengthRepeatCode;
        const fill_value: u8 = if (use_prev) prev_code_len else 0;
        for (0..repeat) |_| {
            code_lengths[emitted] = fill_value;
            emitted += 1;
        }
    }

    var non_zero_count: usize = 0;
    var preview = [_]u8{0} ** 32;
    const preview_len = @min(preview.len, alphabet_size);
    for (0..alphabet_size) |i| {
        const value = code_lengths[i];
        if (value != 0) non_zero_count += 1;
        if (i < preview_len) preview[i] = value;
    }

    return .{
        .decoded_symbol_tokens = tokens,
        .emitted_code_lengths = emitted,
        .non_zero_code_lengths = non_zero_count,
        .preview_len = preview_len,
        .preview = preview,
    };
}

fn parseRuntimePrefixCode(reader: *Vp8lBitReader, alphabet_size: usize) !CanonicalPrefixDecoder {
    if (alphabet_size == 0 or alphabet_size > maxPrefixAlphabetSize) return error.InvalidWebpData;
    const is_simple = (try reader.readBits(1)) == 1;
    var code_lengths = [_]u8{0} ** maxPrefixAlphabetSize;

    if (is_simple) {
        const num_symbols = (try reader.readBits(1)) + 1;
        const is_first_8bits = (try reader.readBits(1)) == 1;
        const symbol0 = try reader.readBits(if (is_first_8bits) 8 else 1);
        if (symbol0 >= alphabet_size) return error.InvalidWebpData;
        code_lengths[symbol0] = 1;
        if (num_symbols == 2) {
            const symbol1 = try reader.readBits(8);
            if (symbol1 >= alphabet_size) return error.InvalidWebpData;
            code_lengths[symbol1] = 1;
        }
        return CanonicalPrefixDecoder.initFromU8(code_lengths[0..alphabet_size]);
    }

    const num_code_length_codes = (try reader.readBits(4)) + 4;
    var code_length_code_lengths = [_]usize{0} ** 19;
    for (0..num_code_length_codes) |i| {
        const symbol = codeLengthCodeOrder[i];
        code_length_code_lengths[symbol] = try reader.readBits(3);
    }

    const use_explicit_max_symbol = (try reader.readBits(1)) == 1;
    const length_nbits = if (use_explicit_max_symbol) 2 + 2 * (try reader.readBits(3)) else null;
    const max_symbol = if (use_explicit_max_symbol) 2 + (try reader.readBits(length_nbits.?)) else alphabet_size;
    if (max_symbol > alphabet_size) return error.InvalidWebpData;

    _ = try inspectDecodedCodeLengths(reader, code_length_code_lengths, alphabet_size, max_symbol, &code_lengths);
    return CanonicalPrefixDecoder.initFromU8(code_lengths[0..alphabet_size]);
}

fn planeCodeToDistanceFast(width: usize, dist_code: usize) usize {
    if (dist_code <= 4) return dist_code;
    const offset = dist_code - 5;
    const row = offset / 12;
    const col = offset % 12;
    const y = @as(isize, @intCast(row / 2 + 1));
    const signed_y: isize = if ((row & 1) == 0) -y else y;
    const x_mag = @as(isize, @intCast(col / 2 + 1));
    const signed_x: isize = if ((col & 1) == 0) -x_mag else x_mag;
    const distance = signed_y * @as(isize, @intCast(width)) + signed_x;
    return @intCast(@max(distance, 1));
}

fn reverseBits(value: usize, bit_count: usize) usize {
    var result: usize = 0;
    for (0..bit_count) |i| {
        result <<= 1;
        result |= (value >> @intCast(i)) & 1;
    }
    return result;
}
