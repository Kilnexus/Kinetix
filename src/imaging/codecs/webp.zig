const std = @import("std");
const types_mod = @import("webp/types.zig");
const bitreader_mod = @import("webp/bitreader.zig");
const container = @import("webp/container.zig");
const probe = @import("webp/probe.zig");
const prefix_codes = @import("webp/prefix_codes.zig");
const color_cache_mod = @import("webp/color_cache.zig");
const transforms_mod = @import("webp/transforms.zig");

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
pub const inspectVp8lNormalPrefixCodeAtBitPos = prefix_codes.inspectVp8lNormalPrefixCodeAtBitPos;
pub const inspectVp8lCanonicalSymbolStreamAtBitPos = prefix_codes.inspectVp8lCanonicalSymbolStreamAtBitPos;
pub const inspectVp8lPrefixCodeGroupAtBitPos = prefix_codes.inspectVp8lPrefixCodeGroupAtBitPos;
const Vp8lBitReader = bitreader_mod.Vp8lBitReader;
const maxPrefixAlphabetSize = prefix_codes.maxPrefixAlphabetSize;
const numPrefixCodes = prefix_codes.numPrefixCodes;
const numLengthCodes = prefix_codes.numLengthCodes;
const numDistanceCodes = prefix_codes.numDistanceCodes;
const RuntimePrefixCodeGroup = prefix_codes.RuntimePrefixCodeGroup;
const CanonicalPrefixDecoder = prefix_codes.CanonicalPrefixDecoder;
const inspectPrefixCodeGroup = prefix_codes.inspectPrefixCodeGroup;
const parseRuntimePrefixCodeGroup = prefix_codes.parseRuntimePrefixCodeGroup;
const readPrefixCodedValue = prefix_codes.readPrefixCodedValue;
const planeCodeToDistance = prefix_codes.planeCodeToDistance;
const packArgb = color_cache_mod.packArgb;
const updateColorCache = color_cache_mod.updateColorCache;
const argbToRgb8 = color_cache_mod.argbToRgb8;
const divRoundUp = color_cache_mod.divRoundUp;
const colorIndexWidthBits = color_cache_mod.colorIndexWidthBits;
const applySupportedTransformsInPlace = transforms_mod.applySupportedTransformsInPlace;
const restoreColorIndexPaletteInPlace = transforms_mod.restoreColorIndexPaletteInPlace;
const expandColorIndexedImage = transforms_mod.expandColorIndexedImage;

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
