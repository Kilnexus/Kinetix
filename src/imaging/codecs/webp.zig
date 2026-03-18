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

pub const Vp8lTransformType = enum {
    predictor,
    color,
    subtract_green,
    color_indexing,
};

pub const Vp8lImageRole = enum {
    argb,
    predictor,
    color,
    color_indexing,
    entropy,
};

pub const Vp8lPrefixCodeKind = enum {
    simple,
    normal,
};

pub const Vp8lSimplePrefixCode = struct {
    num_symbols: usize,
    is_first_8bits: bool,
    symbol0: usize,
    symbol1: ?usize,
    end_bit_pos: usize,
};

pub const Vp8lNormalPrefixCode = struct {
    num_code_length_codes: usize,
    code_length_code_lengths: [19]usize,
    use_explicit_max_symbol: bool,
    length_nbits: ?usize,
    max_symbol: usize,
    decoded_symbol_tokens: ?usize = null,
    emitted_code_lengths: ?usize = null,
    non_zero_code_lengths: ?usize = null,
    preview_len: usize = 0,
    preview: [32]u8 = [_]u8{0} ** 32,
    canonical_summary: ?Vp8lCanonicalPrefixSummary = null,
    end_bit_pos: usize,
};

pub const Vp8lCanonicalCodeEntry = struct {
    symbol: usize,
    len: usize,
    lsb_code: usize,
};

pub const Vp8lCanonicalPrefixSummary = struct {
    active_symbol_count: usize,
    max_code_length: usize,
    preview_len: usize,
    preview: [16]Vp8lCanonicalCodeEntry,
};

pub const Vp8lCanonicalSymbolStream = struct {
    start_bit_pos: usize,
    end_bit_pos: usize,
    symbol_count: usize,
    preview_len: usize,
    preview: [32]usize,
};

pub const Vp8lPrefixCodeHeader = struct {
    kind: Vp8lPrefixCodeKind,
    start_bit_pos: usize,
    simple: ?Vp8lSimplePrefixCode = null,
    normal: ?Vp8lNormalPrefixCode = null,
};

pub const Vp8lPrefixCodeGroup = struct {
    parsed_count: usize,
    all_simple: bool,
    codes: [5]Vp8lPrefixCodeHeader,
};

pub const Vp8lEntropyImageDataHeader = struct {
    width: usize,
    height: usize,
    start_bit_pos: usize,
    use_color_cache: bool,
    color_cache_bits: ?usize,
    header_end_bit_pos: usize,
    prefix_codes_start_bit_pos: usize,
    prefix_group: Vp8lPrefixCodeGroup,
};

pub const Vp8lImageDataHeader = struct {
    role: Vp8lImageRole,
    width: usize,
    height: usize,
    start_bit_pos: usize,
    use_color_cache: bool,
    color_cache_bits: ?usize,
    meta_prefix_present: ?bool,
    prefix_bits: ?usize,
    prefix_image_width: ?usize,
    prefix_image_height: ?usize,
    prefix_image_start_bit_pos: ?usize,
    prefix_image_header: ?Vp8lEntropyImageDataHeader,
    header_end_bit_pos: usize,
    prefix_codes_start_bit_pos: ?usize,
    prefix_group: ?Vp8lPrefixCodeGroup,
};

pub const Vp8lTransform = struct {
    kind: Vp8lTransformType,
    size_bits: ?usize = null,
    color_table_size: ?usize = null,
    width_bits: ?usize = null,
    subimage_start_bit_pos: ?usize = null,
    subimage_width: ?usize = null,
    subimage_height: ?usize = null,
    subimage_header: ?Vp8lImageDataHeader = null,
    transform_width: ?usize = null,
    transform_height: ?usize = null,
    next_image_width: usize,
    next_image_height: usize,
};

pub const Vp8lStreamInfo = struct {
    width: usize,
    height: usize,
    has_alpha: bool,
    header_end_bit_pos: usize,
    image_data_start_bit_pos: ?usize,
    main_image_header: ?Vp8lImageDataHeader,
    transform_count: usize,
    transforms: [4]Vp8lTransform,
    tail_flags_known: bool,
    use_color_cache: ?bool,
    color_cache_bits: ?usize,
    use_meta_prefix: ?bool,
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
    TooManyWebpTransforms,
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

pub fn inspectVp8l(bytes: []const u8) !Vp8lStreamInfo {
    const chunk = try findPrimaryChunk(bytes);
    if (chunk.tag != .vp8l) return error.UnsupportedWebpBitstream;
    return inspectVp8lPayload(chunk.payload);
}

pub fn inspectVp8lImageDataAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    width: usize,
    height: usize,
    role: Vp8lImageRole,
) !Vp8lImageDataHeader {
    return inspectImageDataAtBitPos(payload, start_bit_pos, width, height, role);
}

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

pub fn inspectVp8lPayload(payload: []const u8) !Vp8lStreamInfo {
    const info = try parseVp8l(payload);
    var reader = Vp8lBitReader.init(payload);
    _ = try reader.readBits(8);
    _ = try reader.readBits(14);
    _ = try reader.readBits(14);
    _ = try reader.readBits(1);
    _ = try reader.readBits(3);

    var transforms = [_]Vp8lTransform{undefined} ** 4;
    var transform_count: usize = 0;
    var current_width = info.width;
    const current_height = info.height;

    var tail_flags_known = true;
    while ((try reader.readBits(1)) == 1) {
        if (transform_count >= transforms.len) return error.TooManyWebpTransforms;
        const kind_bits = try reader.readBits(2);
        transforms[transform_count] = switch (kind_bits) {
            0 => blk: {
                tail_flags_known = false;
                const size_bits = (try reader.readBits(3)) + 2;
                const scale = @as(usize, 1) << @intCast(size_bits);
                const transform_width = divRoundUp(current_width, scale);
                const transform_height = divRoundUp(current_height, scale);
                const subimage_start_bit_pos = reader.bit_pos;
                break :blk .{
                    .kind = .predictor,
                    .size_bits = size_bits,
                    .subimage_start_bit_pos = subimage_start_bit_pos,
                    .subimage_width = transform_width,
                    .subimage_height = transform_height,
                    .subimage_header = try inspectImageDataAtBitPos(
                        payload,
                        subimage_start_bit_pos,
                        transform_width,
                        transform_height,
                        .predictor,
                    ),
                    .transform_width = transform_width,
                    .transform_height = transform_height,
                    .next_image_width = current_width,
                    .next_image_height = current_height,
                };
            },
            1 => blk: {
                tail_flags_known = false;
                const size_bits = (try reader.readBits(3)) + 2;
                const scale = @as(usize, 1) << @intCast(size_bits);
                const transform_width = divRoundUp(current_width, scale);
                const transform_height = divRoundUp(current_height, scale);
                const subimage_start_bit_pos = reader.bit_pos;
                break :blk .{
                    .kind = .color,
                    .size_bits = size_bits,
                    .subimage_start_bit_pos = subimage_start_bit_pos,
                    .subimage_width = transform_width,
                    .subimage_height = transform_height,
                    .subimage_header = try inspectImageDataAtBitPos(
                        payload,
                        subimage_start_bit_pos,
                        transform_width,
                        transform_height,
                        .color,
                    ),
                    .transform_width = transform_width,
                    .transform_height = transform_height,
                    .next_image_width = current_width,
                    .next_image_height = current_height,
                };
            },
            2 => .{
                .kind = .subtract_green,
                .next_image_width = current_width,
                .next_image_height = current_height,
            },
            3 => blk: {
                tail_flags_known = false;
                const color_table_size = (try reader.readBits(8)) + 1;
                const width_bits = colorIndexWidthBits(color_table_size);
                current_width = divRoundUp(current_width, @as(usize, 1) << @intCast(width_bits));
                const subimage_start_bit_pos = reader.bit_pos;
                break :blk .{
                    .kind = .color_indexing,
                    .color_table_size = color_table_size,
                    .width_bits = width_bits,
                    .subimage_start_bit_pos = subimage_start_bit_pos,
                    .subimage_width = color_table_size,
                    .subimage_height = 1,
                    .subimage_header = try inspectImageDataAtBitPos(
                        payload,
                        subimage_start_bit_pos,
                        color_table_size,
                        1,
                        .color_indexing,
                    ),
                    .transform_width = color_table_size,
                    .transform_height = 1,
                    .next_image_width = current_width,
                    .next_image_height = current_height,
                };
            },
            else => unreachable,
        };
        transform_count += 1;
        if (!tail_flags_known) break;
    }

    const use_color_cache: ?bool = if (tail_flags_known) (try reader.readBits(1)) == 1 else null;
    const color_cache_bits = if (use_color_cache != null and use_color_cache.?) try reader.readBits(4) else null;
    const use_meta_prefix: ?bool = if (tail_flags_known) (try reader.readBits(1)) == 1 else null;
    const image_data_start_bit_pos = if (tail_flags_known) reader.bit_pos else null;
    const main_image_header = if (tail_flags_known)
        try inspectImageDataAtBitPos(payload, image_data_start_bit_pos.?, current_width, current_height, .argb)
    else
        null;

    return .{
        .width = info.width,
        .height = info.height,
        .has_alpha = info.has_alpha,
        .header_end_bit_pos = reader.bit_pos,
        .image_data_start_bit_pos = image_data_start_bit_pos,
        .main_image_header = main_image_header,
        .transform_count = transform_count,
        .transforms = transforms,
        .tail_flags_known = tail_flags_known,
        .use_color_cache = use_color_cache,
        .color_cache_bits = color_cache_bits,
        .use_meta_prefix = use_meta_prefix,
    };
}

fn inspectImageDataAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    width: usize,
    height: usize,
    role: Vp8lImageRole,
) !Vp8lImageDataHeader {
    var reader = Vp8lBitReader.initAtBit(payload, start_bit_pos);
    const use_color_cache = (try reader.readBits(1)) == 1;
    const color_cache_bits = if (use_color_cache) try reader.readBits(4) else null;

    var meta_prefix_present: ?bool = null;
    var prefix_bits: ?usize = null;
    var prefix_image_width: ?usize = null;
    var prefix_image_height: ?usize = null;
    var prefix_image_start_bit_pos: ?usize = null;
    var prefix_image_header: ?Vp8lEntropyImageDataHeader = null;
    var prefix_codes_start_bit_pos: ?usize = null;
    var prefix_group: ?Vp8lPrefixCodeGroup = null;

    if (role == .argb) {
        meta_prefix_present = (try reader.readBits(1)) == 1;
        if (meta_prefix_present.?) {
            prefix_bits = (try reader.readBits(3)) + 2;
            const scale = @as(usize, 1) << @intCast(prefix_bits.?);
            prefix_image_width = divRoundUp(width, scale);
            prefix_image_height = divRoundUp(height, scale);
            prefix_image_start_bit_pos = reader.bit_pos;
            prefix_image_header = try inspectEntropyImageDataAtBitPos(
                payload,
                prefix_image_start_bit_pos.?,
                prefix_image_width.?,
                prefix_image_height.?,
            );
        }
    }

    if (meta_prefix_present == null or meta_prefix_present.? == false) {
        prefix_codes_start_bit_pos = reader.bit_pos;
        prefix_group = try inspectPrefixCodeGroup(&reader);
    }

    return .{
        .role = role,
        .width = width,
        .height = height,
        .start_bit_pos = start_bit_pos,
        .use_color_cache = use_color_cache,
        .color_cache_bits = color_cache_bits,
        .meta_prefix_present = meta_prefix_present,
        .prefix_bits = prefix_bits,
        .prefix_image_width = prefix_image_width,
        .prefix_image_height = prefix_image_height,
        .prefix_image_start_bit_pos = prefix_image_start_bit_pos,
        .prefix_image_header = prefix_image_header,
        .header_end_bit_pos = reader.bit_pos,
        .prefix_codes_start_bit_pos = prefix_codes_start_bit_pos,
        .prefix_group = prefix_group,
    };
}

fn inspectEntropyImageDataAtBitPos(
    payload: []const u8,
    start_bit_pos: usize,
    width: usize,
    height: usize,
) !Vp8lEntropyImageDataHeader {
    var reader = Vp8lBitReader.initAtBit(payload, start_bit_pos);
    const use_color_cache = (try reader.readBits(1)) == 1;
    const color_cache_bits = if (use_color_cache) try reader.readBits(4) else null;
    const prefix_codes_start_bit_pos = reader.bit_pos;
    const prefix_group = try inspectPrefixCodeGroup(&reader);

    return .{
        .width = width,
        .height = height,
        .start_bit_pos = start_bit_pos,
        .use_color_cache = use_color_cache,
        .color_cache_bits = color_cache_bits,
        .header_end_bit_pos = reader.bit_pos,
        .prefix_codes_start_bit_pos = prefix_codes_start_bit_pos,
        .prefix_group = prefix_group,
    };
}

fn inspectPrefixCodeGroup(reader: *Vp8lBitReader) !Vp8lPrefixCodeGroup {
    var codes = [_]Vp8lPrefixCodeHeader{undefined} ** 5;
    var parsed_count: usize = 0;
    var all_simple = true;

    while (parsed_count < codes.len) : (parsed_count += 1) {
        const start_bit_pos = reader.bit_pos;
        const is_simple = (try reader.readBits(1)) == 1;
        if (!is_simple) {
            all_simple = false;
            const normal = try inspectNormalPrefixCode(reader);
            codes[parsed_count] = .{
                .kind = .normal,
                .start_bit_pos = start_bit_pos,
                .normal = normal,
            };
            parsed_count += 1;
            break;
        }

        const num_symbols = (try reader.readBits(1)) + 1;
        const is_first_8bits = (try reader.readBits(1)) == 1;
        const symbol0 = try reader.readBits(if (is_first_8bits) 8 else 1);
        const symbol1: ?usize = if (num_symbols == 2) try reader.readBits(8) else null;
        codes[parsed_count] = .{
            .kind = .simple,
            .start_bit_pos = start_bit_pos,
            .simple = .{
                .num_symbols = num_symbols,
                .is_first_8bits = is_first_8bits,
                .symbol0 = symbol0,
                .symbol1 = symbol1,
                .end_bit_pos = reader.bit_pos,
            },
        };
    }

    return .{
        .parsed_count = parsed_count,
        .all_simple = all_simple,
        .codes = codes,
    };
}

fn inspectNormalPrefixCode(reader: *Vp8lBitReader) !Vp8lNormalPrefixCode {
    return inspectNormalPrefixCodeImpl(reader, null);
}

fn inspectNormalPrefixCodeDetailed(reader: *Vp8lBitReader, alphabet_size: usize) !Vp8lNormalPrefixCode {
    return inspectNormalPrefixCodeImpl(reader, alphabet_size);
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

fn buildCanonicalPrefixSummary(code_lengths: []const u8) !Vp8lCanonicalPrefixSummary {
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

const Vp8lBitReader = struct {
    bytes: []const u8,
    bit_pos: usize = 0,

    fn init(bytes: []const u8) Vp8lBitReader {
        return .{ .bytes = bytes };
    }

    fn initAtBit(bytes: []const u8, bit_pos: usize) Vp8lBitReader {
        return .{
            .bytes = bytes,
            .bit_pos = bit_pos,
        };
    }

    fn readBits(self: *Vp8lBitReader, count: usize) !usize {
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

const WebpScan = struct {
    info: WebpInfo,
    primary: WebpChunk,
};

const maxPrefixAlphabetSize = 256 + 24 + (1 << 11);
const codeLengthLiteralCount = 16;
const codeLengthRepeatCode = 16;
const defaultCodeLength: u8 = 8;
const codeLengthExtraBits = [3]usize{ 2, 3, 7 };
const codeLengthRepeatOffsets = [3]usize{ 3, 3, 11 };

const codeLengthCodeOrder = [19]usize{
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15,
};

const CanonicalPrefixDecoder = struct {
    const Entry = struct {
        symbol: usize,
        len: usize,
        code: usize,
    };

    entries: [19]Entry = [_]Entry{.{ .symbol = 0, .len = 0, .code = 0 }} ** 19,
    entry_count: usize = 0,
    max_len: usize = 0,

    fn init(code_lengths: []const usize) !CanonicalPrefixDecoder {
        return initImpl(usize, code_lengths);
    }

    fn initFromU8(code_lengths: []const u8) !CanonicalPrefixDecoder {
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

    fn readSymbol(self: *const CanonicalPrefixDecoder, reader: *Vp8lBitReader) !usize {
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

fn divRoundUp(num: usize, den: usize) usize {
    return (num + den - 1) / den;
}

fn colorIndexWidthBits(color_table_size: usize) usize {
    if (color_table_size <= 2) return 3;
    if (color_table_size <= 4) return 2;
    if (color_table_size <= 16) return 1;
    return 0;
}

fn reverseBits(value: usize, bit_count: usize) usize {
    var result: usize = 0;
    for (0..bit_count) |i| {
        result <<= 1;
        result |= (value >> @intCast(i)) & 1;
    }
    return result;
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
