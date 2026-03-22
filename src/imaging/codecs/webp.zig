const std = @import("std");
const types_mod = @import("webp/types.zig");
const bitreader_mod = @import("webp/bitreader.zig");
const container = @import("webp/container.zig");
const probe = @import("webp/probe.zig");

pub const ImageU8 = types_mod.ImageU8;
pub const WebpKind = types_mod.WebpKind;
pub const WebpChunkTag = types_mod.WebpChunkTag;
pub const WebpChunk = types_mod.WebpChunk;
pub const Vp8lTransformType = types_mod.Vp8lTransformType;
pub const Vp8lImageRole = types_mod.Vp8lImageRole;
pub const Vp8lPrefixCodeKind = types_mod.Vp8lPrefixCodeKind;
pub const Vp8lSimplePrefixCode = types_mod.Vp8lSimplePrefixCode;
pub const Vp8lNormalPrefixCode = types_mod.Vp8lNormalPrefixCode;
pub const Vp8lCanonicalCodeEntry = types_mod.Vp8lCanonicalCodeEntry;
pub const Vp8lCanonicalPrefixSummary = types_mod.Vp8lCanonicalPrefixSummary;
pub const Vp8lCanonicalSymbolStream = types_mod.Vp8lCanonicalSymbolStream;
pub const Vp8lPrefixCodeGroupDetail = types_mod.Vp8lPrefixCodeGroupDetail;
pub const Vp8lEventKind = types_mod.Vp8lEventKind;
pub const Vp8lEvent = types_mod.Vp8lEvent;
pub const Vp8lEventStream = types_mod.Vp8lEventStream;
pub const Vp8lArgbImage = types_mod.Vp8lArgbImage;
pub const Vp8lPrefixCodeHeader = types_mod.Vp8lPrefixCodeHeader;
pub const Vp8lPrefixCodeGroup = types_mod.Vp8lPrefixCodeGroup;
pub const Vp8lEntropyImageDataHeader = types_mod.Vp8lEntropyImageDataHeader;
pub const Vp8lImageDataHeader = types_mod.Vp8lImageDataHeader;
pub const Vp8lTransform = types_mod.Vp8lTransform;
pub const Vp8lStreamInfo = types_mod.Vp8lStreamInfo;
pub const WebpInfo = types_mod.WebpInfo;
pub const WebpError = types_mod.WebpError;
pub const ChunkIterator = container.ChunkIterator;
pub const validateHeader = container.validateHeader;
const Vp8lBitReader = bitreader_mod.Vp8lBitReader;

pub fn decodeRgb8(allocator: std.mem.Allocator, bytes: []const u8) !ImageU8 {
    const scan = try probe.scanChunks(bytes);
    if (scan.info.is_animated) return error.UnsupportedWebpAnimation;
    return switch (scan.primary.tag) {
        .vp8 => error.UnsupportedWebpBitstream,
        .vp8l => decodeVp8lRgb8(allocator, scan.primary.payload),
        .vp8x => error.UnsupportedWebpBitstream,
        else => error.MissingWebpChunk,
    };
}

fn decodeVp8lRgb8(allocator: std.mem.Allocator, payload: []const u8) !ImageU8 {
    var argb = try decodeVp8lPayloadArgb(allocator, payload);
    defer argb.deinit();
    return argbToRgb8(allocator, argb.pixels, argb.width, argb.height);
}

pub fn probeInfo(bytes: []const u8) !WebpInfo {
    return probe.probeInfo(bytes);
}

pub fn findPrimaryChunk(bytes: []const u8) !WebpChunk {
    return probe.findPrimaryChunk(bytes);
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

pub fn inspectVp8lEventStreamAtBitPos(
    payload: []const u8,
    prefix_group_start_bit_pos: usize,
    alphabet_sizes: [5]usize,
    width: usize,
    height: usize,
    color_cache_bits: usize,
    max_events: usize,
) !Vp8lEventStream {
    var reader = Vp8lBitReader.initAtBit(payload, prefix_group_start_bit_pos);
    const runtime_group = try parseRuntimePrefixCodeGroup(&reader, alphabet_sizes);
    const event_stream_start_bit_pos = reader.bit_pos;
    const color_cache_size = if (color_cache_bits == 0) 0 else @as(usize, 1) << @intCast(color_cache_bits);
    const max_pixels = width * height;

    var preview = [_]Vp8lEvent{.{ .kind = .literal }} ** 32;
    var preview_len: usize = 0;
    var event_count: usize = 0;
    var emitted_pixels: usize = 0;

    while (emitted_pixels < max_pixels and event_count < max_events) : (event_count += 1) {
        const symbol = try runtime_group.codes[0].readSymbol(&reader);
        var event = Vp8lEvent{ .kind = .literal };
        if (symbol < 256) {
            event.kind = .literal;
            event.green = @intCast(symbol);
            event.red = @intCast(try runtime_group.codes[1].readSymbol(&reader));
            event.blue = @intCast(try runtime_group.codes[2].readSymbol(&reader));
            event.alpha = @intCast(try runtime_group.codes[3].readSymbol(&reader));
            emitted_pixels += 1;
        } else if (symbol < 256 + numLengthCodes) {
            const length_symbol = symbol - 256;
            const length = try readPrefixCodedValue(length_symbol, &reader);
            const distance_symbol = try runtime_group.codes[4].readSymbol(&reader);
            const distance_code = try readPrefixCodedValue(distance_symbol, &reader);
            const distance = planeCodeToDistance(width, distance_code);
            event.kind = .copy;
            event.length_symbol = length_symbol;
            event.length = length;
            event.distance_symbol = distance_symbol;
            event.distance_code = distance_code;
            event.distance = distance;
            emitted_pixels += length;
        } else {
            const cache_index = symbol - (256 + numLengthCodes);
            if (cache_index >= color_cache_size) return error.InvalidWebpData;
            event.kind = .color_cache;
            event.cache_index = cache_index;
            emitted_pixels += 1;
        }

        if (preview_len < preview.len) {
            preview[preview_len] = event;
            preview_len += 1;
        }
    }

    return .{
        .prefix_group_start_bit_pos = prefix_group_start_bit_pos,
        .event_stream_start_bit_pos = event_stream_start_bit_pos,
        .end_bit_pos = reader.bit_pos,
        .event_count = event_count,
        .emitted_pixels = emitted_pixels,
        .preview_len = preview_len,
        .preview = preview,
    };
}

pub fn resolveMetaPrefixCode(
    entropy_image: ?[]const u32,
    prefix_bits: usize,
    prefix_image_width: usize,
    x: usize,
    y: usize,
) !usize {
    if (entropy_image == null) return 0;
    const image = entropy_image.?;
    const position = (y >> @intCast(prefix_bits)) * prefix_image_width + (x >> @intCast(prefix_bits));
    if (position >= image.len) return error.InvalidWebpData;
    return (image[position] >> 8) & 0xffff;
}

pub fn decodeVp8lSingleGroupArgbAtBitPos(
    allocator: std.mem.Allocator,
    payload: []const u8,
    prefix_group_start_bit_pos: usize,
    alphabet_sizes: [5]usize,
    width: usize,
    height: usize,
    color_cache_bits: usize,
) !Vp8lArgbImage {
    var reader = Vp8lBitReader.initAtBit(payload, prefix_group_start_bit_pos);
    const runtime_group = try parseRuntimePrefixCodeGroup(&reader, alphabet_sizes);
    const pixel_count = width * height;
    const pixels = try allocator.alloc(u32, pixel_count);
    errdefer allocator.free(pixels);

    const color_cache_size = if (color_cache_bits == 0) 0 else @as(usize, 1) << @intCast(color_cache_bits);
    const color_cache = if (color_cache_size == 0) null else try allocator.alloc(u32, color_cache_size);
    defer if (color_cache) |cache| allocator.free(cache);
    if (color_cache) |cache| @memset(cache, 0);

    var written: usize = 0;
    while (written < pixel_count) {
        const symbol = try runtime_group.codes[0].readSymbol(&reader);
        if (symbol < 256) {
            const green: u8 = @intCast(symbol);
            const red: u8 = @intCast(try runtime_group.codes[1].readSymbol(&reader));
            const blue: u8 = @intCast(try runtime_group.codes[2].readSymbol(&reader));
            const alpha: u8 = @intCast(try runtime_group.codes[3].readSymbol(&reader));
            const pixel = packArgb(alpha, red, green, blue);
            pixels[written] = pixel;
            updateColorCache(color_cache, color_cache_bits, pixel);
            written += 1;
            continue;
        }

        if (symbol < 256 + numLengthCodes) {
            const length_symbol = symbol - 256;
            const length = try readPrefixCodedValue(length_symbol, &reader);
            const distance_symbol = try runtime_group.codes[4].readSymbol(&reader);
            const distance_code = try readPrefixCodedValue(distance_symbol, &reader);
            const distance = planeCodeToDistance(width, distance_code);
            if (distance == 0 or distance > written) return error.InvalidWebpData;
            if (written + length > pixel_count) return error.InvalidWebpData;
            for (0..length) |_| {
                const pixel = pixels[written - distance];
                pixels[written] = pixel;
                updateColorCache(color_cache, color_cache_bits, pixel);
                written += 1;
            }
            continue;
        }

        const cache_index = symbol - (256 + numLengthCodes);
        if (cache_index >= color_cache_size or color_cache == null) return error.InvalidWebpData;
        const pixel = color_cache.?[cache_index];
        pixels[written] = pixel;
        updateColorCache(color_cache, color_cache_bits, pixel);
        written += 1;
    }

    return .{
        .allocator = allocator,
        .width = width,
        .height = height,
        .end_bit_pos = reader.bit_pos,
        .pixels = pixels,
    };
}

pub fn inspectVp8lPayload(payload: []const u8) !Vp8lStreamInfo {
    const info = try probe.parseVp8l(payload);
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

pub fn decodeVp8lPayloadArgb(allocator: std.mem.Allocator, payload: []const u8) !Vp8lArgbImage {
    const info = try inspectVp8lPayload(payload);
    if (info.tail_flags_known) {
        if (info.main_image_header == null) return error.InvalidWebpData;
        var image = try decodeVp8lImageDataSingleGroupArgb(
            allocator,
            payload,
            info.main_image_header.?,
            info.width,
            info.height,
        );
        errdefer image.deinit();
        try applySupportedTransformsInPlace(&image, info.transforms[0..info.transform_count]);
        return image;
    }

    if (info.transform_count == 1 and info.transforms[0].kind == .color_indexing) {
        return decodeVp8lColorIndexedPayloadArgb(allocator, payload, info, info.transforms[0]);
    }

    return error.UnsupportedWebpBitstream;
}

fn decodeVp8lImageDataSingleGroupArgb(
    allocator: std.mem.Allocator,
    payload: []const u8,
    header: Vp8lImageDataHeader,
    width: usize,
    height: usize,
) !Vp8lArgbImage {
    if (header.meta_prefix_present != null and header.meta_prefix_present.?) return error.UnsupportedWebpBitstream;
    if (header.prefix_codes_start_bit_pos == null) return error.InvalidWebpData;

    const cache_bits = header.color_cache_bits orelse 0;
    const green_alphabet_size = 256 + numLengthCodes + if (cache_bits == 0) @as(usize, 0) else (@as(usize, 1) << @intCast(cache_bits));
    return decodeVp8lSingleGroupArgbAtBitPos(
        allocator,
        payload,
        header.prefix_codes_start_bit_pos.?,
        .{ green_alphabet_size, 256, 256, 256, numDistanceCodes },
        width,
        height,
        cache_bits,
    );
}

fn decodeVp8lColorIndexedPayloadArgb(
    allocator: std.mem.Allocator,
    payload: []const u8,
    info: Vp8lStreamInfo,
    transform: Vp8lTransform,
) !Vp8lArgbImage {
    const palette_header = transform.subimage_header orelse return error.InvalidWebpData;
    const width_bits = transform.width_bits orelse return error.InvalidWebpData;

    var palette_image = try decodeVp8lImageDataSingleGroupArgb(
        allocator,
        payload,
        palette_header,
        palette_header.width,
        palette_header.height,
    );
    defer palette_image.deinit();
    restoreColorIndexPaletteInPlace(palette_image.pixels);

    const encoded_width = transform.next_image_width;
    const encoded_height = transform.next_image_height;
    var indexed_image = try decodeColorIndexedMainImageArgb(
        allocator,
        payload,
        palette_image.end_bit_pos,
        encoded_width,
        encoded_height,
    );
    defer indexed_image.deinit();

    return expandColorIndexedImage(
        allocator,
        indexed_image.pixels,
        palette_image.pixels,
        width_bits,
        info.width,
        info.height,
        indexed_image.end_bit_pos,
    );
}

fn decodeColorIndexedMainImageArgb(
    allocator: std.mem.Allocator,
    payload: []const u8,
    palette_end_bit_pos: usize,
    encoded_width: usize,
    encoded_height: usize,
) !Vp8lArgbImage {
    const roles = [_]Vp8lImageRole{ .argb, .color_indexing };
    const bit_limit = payload.len * 8;

    for (0..17) |offset| {
        const start_bit_pos = palette_end_bit_pos + offset;
        if (start_bit_pos >= bit_limit) continue;
        for (roles) |role| {
            const header = inspectImageDataAtBitPos(payload, start_bit_pos, encoded_width, encoded_height, role) catch continue;
            if (header.prefix_codes_start_bit_pos == null) continue;
            const cache_bits = header.color_cache_bits orelse 0;
            const green_alphabet_size = 256 + numLengthCodes + if (cache_bits == 0) @as(usize, 0) else (@as(usize, 1) << @intCast(cache_bits));
            const stream = inspectVp8lEventStreamAtBitPos(
                payload,
                header.prefix_codes_start_bit_pos.?,
                .{ green_alphabet_size, 256, 256, 256, numDistanceCodes },
                encoded_width,
                encoded_height,
                cache_bits,
                8,
            ) catch continue;
            _ = stream;

            const decoded = decodeVp8lImageDataSingleGroupArgb(
                allocator,
                payload,
                header,
                encoded_width,
                encoded_height,
            ) catch continue;
            return decoded;
        }
    }

    return error.UnsupportedWebpBitstream;
}

fn restoreColorIndexPaletteInPlace(pixels: []u32) void {
    var prev = packArgb(0, 0, 0, 0);
    for (pixels) |*pixel_ptr| {
        const pixel = pixel_ptr.*;
        const restored = packArgb(
            @intCast((((pixel >> 24) & 0xff) + ((prev >> 24) & 0xff)) & 0xff),
            @intCast((((pixel >> 16) & 0xff) + ((prev >> 16) & 0xff)) & 0xff),
            @intCast((((pixel >> 8) & 0xff) + ((prev >> 8) & 0xff)) & 0xff),
            @intCast(((pixel & 0xff) + (prev & 0xff)) & 0xff),
        );
        pixel_ptr.* = restored;
        prev = restored;
    }
}

fn expandColorIndexedImage(
    allocator: std.mem.Allocator,
    indexed_pixels: []const u32,
    palette: []const u32,
    width_bits: usize,
    output_width: usize,
    output_height: usize,
    end_bit_pos: usize,
) !Vp8lArgbImage {
    const pixels_per_index_byte = @as(usize, 1) << @intCast(width_bits);
    const bits_per_index = 8 / pixels_per_index_byte;
    const index_mask = (@as(usize, 1) << @intCast(bits_per_index)) - 1;
    const output_len = output_width * output_height;

    const pixels = try allocator.alloc(u32, output_len);
    errdefer allocator.free(pixels);

    var written: usize = 0;
    for (indexed_pixels) |pixel| {
        const packed_green = @as(usize, (pixel >> 8) & 0xff);
        for (0..pixels_per_index_byte) |slot| {
            if (written >= output_len) break;
            const palette_index = (packed_green >> @intCast(slot * bits_per_index)) & index_mask;
            if (palette_index >= palette.len) return error.InvalidWebpData;
            pixels[written] = palette[palette_index];
            written += 1;
        }
    }
    if (written != output_len) return error.InvalidWebpData;

    return .{
        .allocator = allocator,
        .width = output_width,
        .height = output_height,
        .end_bit_pos = end_bit_pos,
        .pixels = pixels,
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
    return inspectPrefixCodeGroupImpl(reader, null);
}

fn inspectPrefixCodeGroupDetailed(reader: *Vp8lBitReader, alphabet_sizes: [5]usize) !Vp8lPrefixCodeGroup {
    return inspectPrefixCodeGroupImpl(reader, alphabet_sizes);
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

const maxPrefixAlphabetSize = 256 + 24 + (1 << 11);
const codeLengthLiteralCount = 16;
const codeLengthRepeatCode = 16;
const defaultCodeLength: u8 = 8;
const codeLengthExtraBits = [3]usize{ 2, 3, 7 };
const codeLengthRepeatOffsets = [3]usize{ 3, 3, 11 };
const numPrefixCodes = 5;
const numLengthCodes = 24;
const numDistanceCodes = 40;

const codeLengthCodeOrder = [19]usize{
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15,
};

const RuntimePrefixCodeGroup = struct {
    codes: [numPrefixCodes]CanonicalPrefixDecoder,
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

fn parseRuntimePrefixCodeGroup(reader: *Vp8lBitReader, alphabet_sizes: [numPrefixCodes]usize) !RuntimePrefixCodeGroup {
    var decoders = [_]CanonicalPrefixDecoder{undefined} ** numPrefixCodes;
    for (alphabet_sizes, 0..) |alphabet_size, i| {
        decoders[i] = try parseRuntimePrefixCode(reader, alphabet_size);
    }
    return .{ .codes = decoders };
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

fn readPrefixCodedValue(symbol: usize, reader: *Vp8lBitReader) !usize {
    if (symbol < 4) return symbol + 1;
    const extra_bits = (symbol - 2) >> 1;
    const offset = ((2 + (symbol & 1)) << @intCast(extra_bits)) + 1;
    return offset + (try reader.readBits(extra_bits));
}

fn planeCodeToDistance(width: usize, dist_code: usize) usize {
    if (dist_code <= 120) return planeCodeToDistanceFast(width, dist_code);
    return dist_code - 120;
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

fn packArgb(alpha: u8, red: u8, green: u8, blue: u8) u32 {
    return (@as(u32, alpha) << 24) |
        (@as(u32, red) << 16) |
        (@as(u32, green) << 8) |
        @as(u32, blue);
}

fn updateColorCache(color_cache: ?[]u32, color_cache_bits: usize, pixel: u32) void {
    if (color_cache == null or color_cache_bits == 0) return;
    const index = (@as(usize, 0x1e35a7bd) * @as(usize, pixel)) >> @intCast(32 - color_cache_bits);
    color_cache.?[index] = pixel;
}

fn applySupportedTransformsInPlace(image: *Vp8lArgbImage, transforms: []const Vp8lTransform) !void {
    var i = transforms.len;
    while (i > 0) {
        i -= 1;
        switch (transforms[i].kind) {
            .subtract_green => applySubtractGreenTransformInPlace(image.pixels),
            else => return error.UnsupportedWebpBitstream,
        }
    }
}

fn applySubtractGreenTransformInPlace(pixels: []u32) void {
    for (pixels) |*pixel_ptr| {
        const pixel = pixel_ptr.*;
        const alpha = (pixel >> 24) & 0xff;
        const red_delta = (pixel >> 16) & 0xff;
        const green = (pixel >> 8) & 0xff;
        const blue_delta = pixel & 0xff;
        const red = (red_delta + green) & 0xff;
        const blue = (blue_delta + green) & 0xff;
        pixel_ptr.* = (@as(u32, alpha) << 24) |
            (@as(u32, red) << 16) |
            (@as(u32, green) << 8) |
            @as(u32, blue);
    }
}

fn argbToRgb8(allocator: std.mem.Allocator, pixels: []const u32, width: usize, height: usize) !ImageU8 {
    var image = try ImageU8.init(allocator, width, height, 3);
    errdefer image.deinit();
    for (pixels, 0..) |pixel, i| {
        image.data[i * 3] = @intCast((pixel >> 16) & 0xff);
        image.data[i * 3 + 1] = @intCast((pixel >> 8) & 0xff);
        image.data[i * 3 + 2] = @intCast(pixel & 0xff);
    }
    return image;
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
