const imaging = @import("imaging");
const helpers = @import("helpers.zig");
const writeU32le = helpers.writeU32le;
const writeVp8lHeader = helpers.writeVp8lHeader;
const writeBit = helpers.writeBit;
const writeBits = helpers.writeBits;
comptime {
    _ = @import("basic_tests.zig");
    _ = @import("probe_tests.zig");
    _ = @import("codec_decode_tests.zig");
    _ = @import("webp_decode_tests.zig");
}

test "inspectWebpVp8l parses transform chain for lossless samples" {
    const std = @import("std");
    const testing = std.testing;

    const rgb_lossless_base64 = "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=";
    const rgba_lossless_base64 = "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAEAcQ/Y8CBiKi/wEA";

    const rgb_len = try std.base64.standard.Decoder.calcSizeForSlice(rgb_lossless_base64);
    const rgb = try testing.allocator.alloc(u8, rgb_len);
    defer testing.allocator.free(rgb);
    try std.base64.standard.Decoder.decode(rgb, rgb_lossless_base64);

    const rgba_len = try std.base64.standard.Decoder.calcSizeForSlice(rgba_lossless_base64);
    const rgba = try testing.allocator.alloc(u8, rgba_len);
    defer testing.allocator.free(rgba);
    try std.base64.standard.Decoder.decode(rgba, rgba_lossless_base64);

    const rgb_info = try imaging.inspectWebpVp8l(rgb);
    try testing.expectEqual(@as(usize, 2), rgb_info.width);
    try testing.expectEqual(@as(usize, 1), rgb_info.height);
    try testing.expect(!rgb_info.has_alpha);
    try testing.expectEqual(@as(usize, 51), rgb_info.header_end_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgb_info.transform_count);
    try testing.expectEqual(imaging.Vp8lTransformType.color_indexing, rgb_info.transforms[0].kind);
    try testing.expectEqual(@as(?usize, 2), rgb_info.transforms[0].color_table_size);
    try testing.expectEqual(@as(?usize, 3), rgb_info.transforms[0].width_bits);
    try testing.expectEqual(@as(?usize, 51), rgb_info.transforms[0].subimage_start_bit_pos);
    try testing.expectEqual(@as(?usize, 2), rgb_info.transforms[0].subimage_width);
    try testing.expectEqual(@as(?usize, 1), rgb_info.transforms[0].subimage_height);
    try testing.expect(rgb_info.transforms[0].subimage_header != null);
    try testing.expectEqual(imaging.Vp8lImageRole.color_indexing, rgb_info.transforms[0].subimage_header.?.role);
    try testing.expectEqual(@as(usize, 2), rgb_info.transforms[0].subimage_header.?.width);
    try testing.expectEqual(@as(usize, 1), rgb_info.transforms[0].subimage_header.?.height);
    try testing.expectEqual(@as(usize, 51), rgb_info.transforms[0].subimage_header.?.start_bit_pos);
    try testing.expect(!rgb_info.transforms[0].subimage_header.?.use_color_cache);
    try testing.expectEqual(@as(?usize, null), rgb_info.transforms[0].subimage_header.?.color_cache_bits);
    try testing.expectEqual(@as(?bool, null), rgb_info.transforms[0].subimage_header.?.meta_prefix_present);
    try testing.expectEqual(@as(?usize, null), rgb_info.transforms[0].subimage_header.?.prefix_image_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lEntropyImageDataHeader, null), rgb_info.transforms[0].subimage_header.?.prefix_image_header);
    try testing.expectEqual(@as(?usize, 52), rgb_info.transforms[0].subimage_header.?.prefix_codes_start_bit_pos);
    try testing.expectEqual(@as(usize, 96), rgb_info.transforms[0].subimage_header.?.header_end_bit_pos);
    try testing.expect(rgb_info.transforms[0].subimage_header.?.prefix_group != null);
    const rgb_group = rgb_info.transforms[0].subimage_header.?.prefix_group.?;
    try testing.expectEqual(@as(usize, 5), rgb_group.parsed_count);
    try testing.expect(rgb_group.all_simple);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgb_group.codes[0].kind);
    try testing.expectEqual(@as(usize, 52), rgb_group.codes[0].start_bit_pos);
    try testing.expectEqual(@as(usize, 2), rgb_group.codes[0].simple.?.num_symbols);
    try testing.expect(!rgb_group.codes[0].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 1), rgb_group.codes[0].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, 255), rgb_group.codes[0].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 64), rgb_group.codes[0].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgb_group.codes[1].kind);
    try testing.expectEqual(@as(usize, 64), rgb_group.codes[1].start_bit_pos);
    try testing.expectEqual(@as(usize, 2), rgb_group.codes[1].simple.?.num_symbols);
    try testing.expectEqual(@as(usize, 0), rgb_group.codes[1].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, 255), rgb_group.codes[1].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 76), rgb_group.codes[1].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgb_group.codes[2].kind);
    try testing.expectEqual(@as(usize, 76), rgb_group.codes[2].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgb_group.codes[2].simple.?.num_symbols);
    try testing.expectEqual(@as(usize, 0), rgb_group.codes[2].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgb_group.codes[2].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 80), rgb_group.codes[2].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgb_group.codes[3].kind);
    try testing.expectEqual(@as(usize, 80), rgb_group.codes[3].start_bit_pos);
    try testing.expectEqual(@as(usize, 2), rgb_group.codes[3].simple.?.num_symbols);
    try testing.expectEqual(@as(usize, 0), rgb_group.codes[3].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, 255), rgb_group.codes[3].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 92), rgb_group.codes[3].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgb_group.codes[4].kind);
    try testing.expectEqual(@as(usize, 92), rgb_group.codes[4].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgb_group.codes[4].simple.?.num_symbols);
    try testing.expectEqual(@as(usize, 0), rgb_group.codes[4].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgb_group.codes[4].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 96), rgb_group.codes[4].simple.?.end_bit_pos);
    try testing.expectEqual(@as(?usize, 2), rgb_info.transforms[0].transform_width);
    try testing.expectEqual(@as(?usize, 1), rgb_info.transforms[0].transform_height);
    try testing.expectEqual(@as(usize, 1), rgb_info.transforms[0].next_image_width);
    try testing.expectEqual(@as(usize, 1), rgb_info.transforms[0].next_image_height);
    try testing.expect(!rgb_info.tail_flags_known);
    try testing.expectEqual(@as(?usize, null), rgb_info.image_data_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lImageDataHeader, null), rgb_info.main_image_header);
    try testing.expectEqual(@as(?bool, null), rgb_info.use_color_cache);
    try testing.expectEqual(@as(?bool, null), rgb_info.use_meta_prefix);

    const rgba_info = try imaging.inspectWebpVp8l(rgba);
    try testing.expectEqual(@as(usize, 1), rgba_info.width);
    try testing.expectEqual(@as(usize, 1), rgba_info.height);
    try testing.expect(rgba_info.has_alpha);
    try testing.expectEqual(@as(usize, 51), rgba_info.header_end_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_info.transform_count);
    try testing.expectEqual(imaging.Vp8lTransformType.color_indexing, rgba_info.transforms[0].kind);
    try testing.expectEqual(@as(?usize, 1), rgba_info.transforms[0].color_table_size);
    try testing.expectEqual(@as(?usize, 3), rgba_info.transforms[0].width_bits);
    try testing.expectEqual(@as(?usize, 51), rgba_info.transforms[0].subimage_start_bit_pos);
    try testing.expectEqual(@as(?usize, 1), rgba_info.transforms[0].subimage_width);
    try testing.expectEqual(@as(?usize, 1), rgba_info.transforms[0].subimage_height);
    try testing.expect(rgba_info.transforms[0].subimage_header != null);
    try testing.expectEqual(imaging.Vp8lImageRole.color_indexing, rgba_info.transforms[0].subimage_header.?.role);
    try testing.expectEqual(@as(usize, 1), rgba_info.transforms[0].subimage_header.?.width);
    try testing.expectEqual(@as(usize, 1), rgba_info.transforms[0].subimage_header.?.height);
    try testing.expectEqual(@as(usize, 51), rgba_info.transforms[0].subimage_header.?.start_bit_pos);
    try testing.expect(!rgba_info.transforms[0].subimage_header.?.use_color_cache);
    try testing.expectEqual(@as(?usize, null), rgba_info.transforms[0].subimage_header.?.color_cache_bits);
    try testing.expectEqual(@as(?bool, null), rgba_info.transforms[0].subimage_header.?.meta_prefix_present);
    try testing.expectEqual(@as(?usize, null), rgba_info.transforms[0].subimage_header.?.prefix_image_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lEntropyImageDataHeader, null), rgba_info.transforms[0].subimage_header.?.prefix_image_header);
    try testing.expectEqual(@as(?usize, 52), rgba_info.transforms[0].subimage_header.?.prefix_codes_start_bit_pos);
    try testing.expectEqual(@as(usize, 86), rgba_info.transforms[0].subimage_header.?.header_end_bit_pos);
    try testing.expect(rgba_info.transforms[0].subimage_header.?.prefix_group != null);
    const rgba_group = rgba_info.transforms[0].subimage_header.?.prefix_group.?;
    try testing.expectEqual(@as(usize, 5), rgba_group.parsed_count);
    try testing.expect(rgba_group.all_simple);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgba_group.codes[0].kind);
    try testing.expectEqual(@as(usize, 52), rgba_group.codes[0].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_group.codes[0].simple.?.num_symbols);
    try testing.expect(!rgba_group.codes[0].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 0), rgba_group.codes[0].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgba_group.codes[0].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 56), rgba_group.codes[0].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgba_group.codes[1].kind);
    try testing.expectEqual(@as(usize, 56), rgba_group.codes[1].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_group.codes[1].simple.?.num_symbols);
    try testing.expect(rgba_group.codes[1].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 255), rgba_group.codes[1].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgba_group.codes[1].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 67), rgba_group.codes[1].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgba_group.codes[2].kind);
    try testing.expectEqual(@as(usize, 67), rgba_group.codes[2].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_group.codes[2].simple.?.num_symbols);
    try testing.expect(!rgba_group.codes[2].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 0), rgba_group.codes[2].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgba_group.codes[2].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 71), rgba_group.codes[2].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgba_group.codes[3].kind);
    try testing.expectEqual(@as(usize, 71), rgba_group.codes[3].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_group.codes[3].simple.?.num_symbols);
    try testing.expect(rgba_group.codes[3].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 128), rgba_group.codes[3].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgba_group.codes[3].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 82), rgba_group.codes[3].simple.?.end_bit_pos);
    try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, rgba_group.codes[4].kind);
    try testing.expectEqual(@as(usize, 82), rgba_group.codes[4].start_bit_pos);
    try testing.expectEqual(@as(usize, 1), rgba_group.codes[4].simple.?.num_symbols);
    try testing.expect(!rgba_group.codes[4].simple.?.is_first_8bits);
    try testing.expectEqual(@as(usize, 0), rgba_group.codes[4].simple.?.symbol0);
    try testing.expectEqual(@as(?usize, null), rgba_group.codes[4].simple.?.symbol1);
    try testing.expectEqual(@as(usize, 86), rgba_group.codes[4].simple.?.end_bit_pos);
    try testing.expectEqual(@as(?usize, 1), rgba_info.transforms[0].transform_width);
    try testing.expectEqual(@as(?usize, 1), rgba_info.transforms[0].transform_height);
    try testing.expectEqual(@as(usize, 1), rgba_info.transforms[0].next_image_width);
    try testing.expectEqual(@as(usize, 1), rgba_info.transforms[0].next_image_height);
    try testing.expect(!rgba_info.tail_flags_known);
    try testing.expectEqual(@as(?usize, null), rgba_info.image_data_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lImageDataHeader, null), rgba_info.main_image_header);
    try testing.expectEqual(@as(?bool, null), rgba_info.use_color_cache);
    try testing.expectEqual(@as(?bool, null), rgba_info.use_meta_prefix);
}

test "inspectWebpVp8l parses main image header for simple lossless samples" {
    const std = @import("std");
    const testing = std.testing;

    const solid_red_base64 = "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAAAcQ/Y/+ByKi/wEA";
    const checker_base64 = "UklGRiYAAABXRUJQVlA4TBoAAAAvA8AAAA8w//M///MfeFDTtgGLr6Qjov/BOQ==";
    const gradient_base64 = "UklGRrAAAABXRUJQVlA4TKMAAAAvD8ADEE1kRP9jEYUf8P5HAUHbtjGE8Ke7q6cwEIwhSRJ0GIUyKIuyKItyKIOSTwFJ0vPwuSK3bZtjdtln8LEts6NYObSTOI+5yLhbHrab8Wy5bE/G3QpaiXcl2pto5aWVxJaxfRnb8rEt49sydiTOyp92Ev+VQ/snzirYbsbdco25WIneUWR+8PBRO0l8d6c9urvTHu7utMd3d9qjrX7+7QMXAA==";

    const samples = [_]struct {
        name: []const u8,
        base64: []const u8,
    }{
        .{ .name = "solid_red_1x1", .base64 = solid_red_base64 },
        .{ .name = "checker_4x4", .base64 = checker_base64 },
        .{ .name = "gradient_16x16", .base64 = gradient_base64 },
    };

    for (samples) |sample| {
        const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(sample.base64);
        const webp = try testing.allocator.alloc(u8, decoded_len);
        defer testing.allocator.free(webp);
        try std.base64.standard.Decoder.decode(webp, sample.base64);

        const info = try imaging.inspectWebpVp8l(webp);
        _ = sample.name;
        try testing.expect(!info.tail_flags_known);
        try testing.expectEqual(@as(?usize, null), info.image_data_start_bit_pos);
        try testing.expectEqual(@as(?imaging.Vp8lImageDataHeader, null), info.main_image_header);
    }
}

test "inspectVp8lImageDataAtBitPos parses argb meta prefix branch" {
    const testing = @import("std").testing;

    var payload = [_]u8{0} ** 4;
    var bit_pos: usize = 0;

    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 3);

    writeBit(&payload, &bit_pos, 0);
    for (0..5) |_| {
        writeBit(&payload, &bit_pos, 1);
        writeBit(&payload, &bit_pos, 0);
        writeBit(&payload, &bit_pos, 0);
        writeBit(&payload, &bit_pos, 0);
    }

    const header = try imaging.inspectVp8lImageDataAtBitPos(&payload, 0, 5, 4, .argb);
    try testing.expectEqual(imaging.Vp8lImageRole.argb, header.role);
    try testing.expectEqual(@as(usize, 5), header.width);
    try testing.expectEqual(@as(usize, 4), header.height);
    try testing.expect(!header.use_color_cache);
    try testing.expectEqual(@as(?usize, null), header.color_cache_bits);
    try testing.expectEqual(@as(?bool, true), header.meta_prefix_present);
    try testing.expectEqual(@as(?usize, 2), header.prefix_bits);
    try testing.expectEqual(@as(?usize, 2), header.prefix_image_width);
    try testing.expectEqual(@as(?usize, 1), header.prefix_image_height);
    try testing.expectEqual(@as(?usize, 5), header.prefix_image_start_bit_pos);
    try testing.expectEqual(@as(usize, 5), header.header_end_bit_pos);
    try testing.expectEqual(@as(?usize, null), header.prefix_codes_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lPrefixCodeGroup, null), header.prefix_group);
    try testing.expect(header.prefix_image_header != null);

    const prefix_header = header.prefix_image_header.?;
    try testing.expectEqual(@as(usize, 2), prefix_header.width);
    try testing.expectEqual(@as(usize, 1), prefix_header.height);
    try testing.expectEqual(@as(usize, 5), prefix_header.start_bit_pos);
    try testing.expect(!prefix_header.use_color_cache);
    try testing.expectEqual(@as(?usize, null), prefix_header.color_cache_bits);
    try testing.expectEqual(@as(usize, 6), prefix_header.prefix_codes_start_bit_pos);
    try testing.expectEqual(@as(usize, 26), prefix_header.header_end_bit_pos);
    try testing.expectEqual(@as(usize, 5), prefix_header.prefix_group.parsed_count);
    try testing.expect(prefix_header.prefix_group.all_simple);

    for (0..5) |i| {
        try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, prefix_header.prefix_group.codes[i].kind);
        try testing.expectEqual(@as(usize, 6 + i * 4), prefix_header.prefix_group.codes[i].start_bit_pos);
        try testing.expectEqual(@as(usize, 1), prefix_header.prefix_group.codes[i].simple.?.num_symbols);
        try testing.expect(!prefix_header.prefix_group.codes[i].simple.?.is_first_8bits);
        try testing.expectEqual(@as(usize, 0), prefix_header.prefix_group.codes[i].simple.?.symbol0);
        try testing.expectEqual(@as(?usize, null), prefix_header.prefix_group.codes[i].simple.?.symbol1);
        try testing.expectEqual(@as(usize, 10 + i * 4), prefix_header.prefix_group.codes[i].simple.?.end_bit_pos);
    }
}

test "inspectVp8lImageDataAtBitPos parses prefix code header envelope" {
    const testing = @import("std").testing;

    var payload = [_]u8{0} ** 4;
    var bit_pos: usize = 0;

    writeBit(&payload, &bit_pos, 0);
    for (0..5) |i| {
        writeBit(&payload, &bit_pos, 1);
        writeBit(&payload, &bit_pos, 0);
        writeBit(&payload, &bit_pos, 0);
        writeBit(&payload, &bit_pos, @intCast(i & 1));
    }

    const header = try imaging.inspectVp8lImageDataAtBitPos(&payload, 0, 3, 2, .color);
    try testing.expectEqual(imaging.Vp8lImageRole.color, header.role);
    try testing.expectEqual(@as(usize, 3), header.width);
    try testing.expectEqual(@as(usize, 2), header.height);
    try testing.expect(!header.use_color_cache);
    try testing.expectEqual(@as(?usize, null), header.color_cache_bits);
    try testing.expectEqual(@as(?bool, null), header.meta_prefix_present);
    try testing.expectEqual(@as(?usize, 1), header.prefix_codes_start_bit_pos);
    try testing.expectEqual(@as(usize, bit_pos), header.header_end_bit_pos);
    try testing.expectEqual(@as(?usize, null), header.prefix_image_start_bit_pos);
    try testing.expectEqual(@as(?imaging.Vp8lEntropyImageDataHeader, null), header.prefix_image_header);
    try testing.expect(header.prefix_group != null);

    const group = header.prefix_group.?;
    try testing.expectEqual(@as(usize, 5), group.parsed_count);
    try testing.expect(group.all_simple);
    for (0..5) |i| {
        try testing.expectEqual(imaging.Vp8lPrefixCodeKind.simple, group.codes[i].kind);
        try testing.expect(group.codes[i].simple != null);
    }
}

test "inspectVp8lNormalPrefixCodeAtBitPos decodes literal code length sequence" {
    const testing = @import("std").testing;

    var payload = [_]u8{0} ** 4;
    var bit_pos: usize = 0;

    writeBits(&payload, &bit_pos, 0, 4);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 1, 3);
    writeBits(&payload, &bit_pos, 1, 3);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 2, 2);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 0);

    const normal = try imaging.inspectVp8lNormalPrefixCodeAtBitPos(&payload, 0, 8);
    try testing.expectEqual(@as(usize, 4), normal.num_code_length_codes);
    try testing.expect(normal.use_explicit_max_symbol);
    try testing.expectEqual(@as(?usize, 2), normal.length_nbits);
    try testing.expectEqual(@as(usize, 4), normal.max_symbol);
    try testing.expectEqual(@as(?usize, 4), normal.decoded_symbol_tokens);
    try testing.expectEqual(@as(?usize, 4), normal.emitted_code_lengths);
    try testing.expectEqual(@as(?usize, 2), normal.non_zero_code_lengths);
    try testing.expectEqual(@as(usize, 8), normal.preview_len);
    try testing.expectEqual(@as(usize, 26), normal.end_bit_pos);
    try testing.expect(normal.canonical_summary != null);
    try testing.expectEqual(@as(usize, 0), normal.code_length_code_lengths[17]);
    try testing.expectEqual(@as(usize, 0), normal.code_length_code_lengths[18]);
    try testing.expectEqual(@as(usize, 1), normal.code_length_code_lengths[0]);
    try testing.expectEqual(@as(usize, 1), normal.code_length_code_lengths[1]);
    try testing.expectEqual(@as(u8, 1), normal.preview[0]);
    try testing.expectEqual(@as(u8, 1), normal.preview[1]);
    try testing.expectEqual(@as(u8, 0), normal.preview[2]);
    try testing.expectEqual(@as(u8, 0), normal.preview[3]);
    try testing.expectEqual(@as(u8, 0), normal.preview[4]);
    try testing.expectEqual(@as(u8, 0), normal.preview[5]);
    try testing.expectEqual(@as(u8, 0), normal.preview[6]);
    try testing.expectEqual(@as(u8, 0), normal.preview[7]);

    const summary = normal.canonical_summary.?;
    try testing.expectEqual(@as(usize, 2), summary.active_symbol_count);
    try testing.expectEqual(@as(usize, 1), summary.max_code_length);
    try testing.expectEqual(@as(usize, 2), summary.preview_len);
    try testing.expectEqual(@as(usize, 0), summary.preview[0].symbol);
    try testing.expectEqual(@as(usize, 1), summary.preview[0].len);
    try testing.expectEqual(@as(usize, 0), summary.preview[0].lsb_code);
    try testing.expectEqual(@as(usize, 1), summary.preview[1].symbol);
    try testing.expectEqual(@as(usize, 1), summary.preview[1].len);
    try testing.expectEqual(@as(usize, 1), summary.preview[1].lsb_code);
}

test "inspectVp8lNormalPrefixCodeAtBitPos decodes repeat code lengths and builds canonical summary" {
    const std = @import("std");
    const testing = std.testing;

    var payload = [_]u8{0} ** 8;
    var bit_pos: usize = 0;

    writeBits(&payload, &bit_pos, 5, 4);
    writeBits(&payload, &bit_pos, 2, 3);
    writeBits(&payload, &bit_pos, 2, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 2, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 2, 3);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 2, 2);

    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 0);

    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 1, 2);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBits(&payload, &bit_pos, 0, 3);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 1, 7);

    const normal = try imaging.inspectVp8lNormalPrefixCodeAtBitPos(&payload, 0, 20);
    try testing.expectEqual(@as(usize, 9), normal.num_code_length_codes);
    try testing.expect(normal.use_explicit_max_symbol);
    try testing.expectEqual(@as(?usize, 2), normal.length_nbits);
    try testing.expectEqual(@as(usize, 4), normal.max_symbol);
    try testing.expectEqual(@as(?usize, 4), normal.decoded_symbol_tokens);
    try testing.expectEqual(@as(?usize, 20), normal.emitted_code_lengths);
    try testing.expectEqual(@as(?usize, 5), normal.non_zero_code_lengths);
    try testing.expectEqual(@as(usize, 20), normal.preview_len);
    try testing.expectEqual(@as(usize, 57), normal.end_bit_pos);
    try testing.expectEqual(@as(usize, 2), normal.code_length_code_lengths[17]);
    try testing.expectEqual(@as(usize, 2), normal.code_length_code_lengths[18]);
    try testing.expectEqual(@as(usize, 0), normal.code_length_code_lengths[0]);
    try testing.expectEqual(@as(usize, 2), normal.code_length_code_lengths[3]);
    try testing.expectEqual(@as(usize, 2), normal.code_length_code_lengths[16]);

    for (0..5) |i| try testing.expectEqual(@as(u8, 3), normal.preview[i]);
    for (5..20) |i| try testing.expectEqual(@as(u8, 0), normal.preview[i]);

    try testing.expect(normal.canonical_summary != null);
    const summary = normal.canonical_summary.?;
    try testing.expectEqual(@as(usize, 5), summary.active_symbol_count);
    try testing.expectEqual(@as(usize, 3), summary.max_code_length);
    try testing.expectEqual(@as(usize, 5), summary.preview_len);
    try testing.expectEqual(@as(usize, 0), summary.preview[0].symbol);
    try testing.expectEqual(@as(usize, 3), summary.preview[0].len);
    try testing.expectEqual(@as(usize, 0), summary.preview[0].lsb_code);
    try testing.expectEqual(@as(usize, 1), summary.preview[1].symbol);
    try testing.expectEqual(@as(usize, 3), summary.preview[1].len);
    try testing.expectEqual(@as(usize, 4), summary.preview[1].lsb_code);
    try testing.expectEqual(@as(usize, 2), summary.preview[2].symbol);
    try testing.expectEqual(@as(usize, 3), summary.preview[2].len);
    try testing.expectEqual(@as(usize, 2), summary.preview[2].lsb_code);
    try testing.expectEqual(@as(usize, 3), summary.preview[3].symbol);
    try testing.expectEqual(@as(usize, 3), summary.preview[3].len);
    try testing.expectEqual(@as(usize, 6), summary.preview[3].lsb_code);
    try testing.expectEqual(@as(usize, 4), summary.preview[4].symbol);
    try testing.expectEqual(@as(usize, 3), summary.preview[4].len);
    try testing.expectEqual(@as(usize, 1), summary.preview[4].lsb_code);
}

test "inspectVp8lCanonicalSymbolStreamAtBitPos decodes 1-bit canonical stream" {
    const testing = @import("std").testing;

    const code_lengths = [_]u8{ 1, 1, 0, 0, 0, 0, 0, 0 };
    var payload = [_]u8{0};
    var bit_pos: usize = 0;
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);

    const stream = try imaging.inspectVp8lCanonicalSymbolStreamAtBitPos(&payload, 0, &code_lengths, 4);
    try testing.expectEqual(@as(usize, 0), stream.start_bit_pos);
    try testing.expectEqual(@as(usize, 4), stream.end_bit_pos);
    try testing.expectEqual(@as(usize, 4), stream.symbol_count);
    try testing.expectEqual(@as(usize, 4), stream.preview_len);
    try testing.expectEqual(@as(usize, 0), stream.preview[0]);
    try testing.expectEqual(@as(usize, 1), stream.preview[1]);
    try testing.expectEqual(@as(usize, 1), stream.preview[2]);
    try testing.expectEqual(@as(usize, 0), stream.preview[3]);
}

test "inspectVp8lCanonicalSymbolStreamAtBitPos handles single-symbol tree without consuming bits" {
    const testing = @import("std").testing;

    const code_lengths = [_]u8{ 0, 0, 1, 0 };
    const payload = [_]u8{0xaa, 0x55};

    const stream = try imaging.inspectVp8lCanonicalSymbolStreamAtBitPos(&payload, 0, &code_lengths, 3);
    try testing.expectEqual(@as(usize, 0), stream.start_bit_pos);
    try testing.expectEqual(@as(usize, 0), stream.end_bit_pos);
    try testing.expectEqual(@as(usize, 3), stream.symbol_count);
    try testing.expectEqual(@as(usize, 3), stream.preview_len);
    try testing.expectEqual(@as(usize, 2), stream.preview[0]);
    try testing.expectEqual(@as(usize, 2), stream.preview[1]);
    try testing.expectEqual(@as(usize, 2), stream.preview[2]);
}

test "inspectVp8lCanonicalSymbolStreamAtBitPos decodes 3-bit canonical stream" {
    const testing = @import("std").testing;

    const code_lengths = [_]u8{
        3, 3, 3, 3, 3,
        0, 0, 0, 0, 0,
    };
    var payload = [_]u8{0} ** 2;
    var bit_pos: usize = 0;

    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 4, 3);
    writeBits(&payload, &bit_pos, 2, 3);
    writeBits(&payload, &bit_pos, 6, 3);
    writeBits(&payload, &bit_pos, 1, 3);

    const stream = try imaging.inspectVp8lCanonicalSymbolStreamAtBitPos(&payload, 0, &code_lengths, 5);
    try testing.expectEqual(@as(usize, 0), stream.start_bit_pos);
    try testing.expectEqual(@as(usize, 15), stream.end_bit_pos);
    try testing.expectEqual(@as(usize, 5), stream.symbol_count);
    try testing.expectEqual(@as(usize, 5), stream.preview_len);
    try testing.expectEqual(@as(usize, 0), stream.preview[0]);
    try testing.expectEqual(@as(usize, 1), stream.preview[1]);
    try testing.expectEqual(@as(usize, 2), stream.preview[2]);
    try testing.expectEqual(@as(usize, 3), stream.preview[3]);
    try testing.expectEqual(@as(usize, 4), stream.preview[4]);
}

test "inspectVp8lPrefixCodeGroupAtBitPos parses mixed group with summaries" {
    const testing = @import("std").testing;

    var payload = [_]u8{0} ** 16;
    var bit_pos: usize = 0;

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 2, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 8);
    writeBits(&payload, &bit_pos, 5, 8);

    writeBits(&payload, &bit_pos, 0, 1);
    writeBits(&payload, &bit_pos, 0, 4);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 1, 3);
    writeBits(&payload, &bit_pos, 1, 3);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 3);
    writeBits(&payload, &bit_pos, 2, 2);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 0);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 1, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 7, 8);

    const detail = try imaging.inspectVp8lPrefixCodeGroupAtBitPos(&payload, 0, .{ 8, 8, 8, 8, 8 });
    try testing.expectEqual(@as(usize, 0), detail.start_bit_pos);
    try testing.expectEqual(bit_pos, detail.end_bit_pos);
    try testing.expectEqual(@as(usize, 5), detail.group.parsed_count);
    try testing.expect(!detail.group.all_simple);

    const code0 = detail.group.codes[0].simple.?;
    try testing.expect(code0.canonical_summary != null);
    try testing.expectEqual(@as(usize, 1), code0.canonical_summary.?.active_symbol_count);
    try testing.expectEqual(@as(usize, 2), code0.canonical_summary.?.preview[0].symbol);

    const code1 = detail.group.codes[1].simple.?;
    try testing.expect(code1.canonical_summary != null);
    try testing.expectEqual(@as(usize, 2), code1.canonical_summary.?.active_symbol_count);
    try testing.expectEqual(@as(usize, 0), code1.canonical_summary.?.preview[0].symbol);
    try testing.expectEqual(@as(usize, 5), code1.canonical_summary.?.preview[1].symbol);

    const code2 = detail.group.codes[2].normal.?;
    try testing.expect(code2.canonical_summary != null);
    try testing.expectEqual(@as(usize, 2), code2.canonical_summary.?.active_symbol_count);
    try testing.expectEqual(@as(usize, 1), code2.canonical_summary.?.max_code_length);

    const code3 = detail.group.codes[3].simple.?;
    try testing.expect(code3.canonical_summary != null);
    try testing.expectEqual(@as(usize, 1), code3.canonical_summary.?.preview[0].symbol);

    const code4 = detail.group.codes[4].simple.?;
    try testing.expect(code4.canonical_summary != null);
    try testing.expectEqual(@as(usize, 7), code4.canonical_summary.?.preview[0].symbol);
}

test "resolveMetaPrefixCode maps source pixel to prefix image group" {
    const testing = @import("std").testing;

    const entropy_image = [_]u32{
        1 << 8, 2 << 8,
        3 << 8, 4 << 8,
    };

    try testing.expectEqual(@as(usize, 1), try imaging.resolveMetaPrefixCode(&entropy_image, 1, 2, 0, 0));
    try testing.expectEqual(@as(usize, 2), try imaging.resolveMetaPrefixCode(&entropy_image, 1, 2, 3, 0));
    try testing.expectEqual(@as(usize, 3), try imaging.resolveMetaPrefixCode(&entropy_image, 1, 2, 0, 3));
    try testing.expectEqual(@as(usize, 4), try imaging.resolveMetaPrefixCode(&entropy_image, 1, 2, 3, 3));
    try testing.expectEqual(@as(usize, 0), try imaging.resolveMetaPrefixCode(null, 1, 2, 3, 3));
}

test "inspectVp8lEventStreamAtBitPos decodes literal-only stream" {
    const testing = @import("std").testing;

    var payload = [_]u8{0} ** 16;
    var bit_pos: usize = 0;

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 1, 8);
    writeBits(&payload, &bit_pos, 2, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 10, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 20, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 255, 8);

    writeBit(&payload, &bit_pos, 1);
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);
    writeBits(&payload, &bit_pos, 0, 8);

    const group_start_bit_pos = 0;
    const event_stream_start_bit_pos = bit_pos;
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);

    const stream = try imaging.inspectVp8lEventStreamAtBitPos(
        &payload,
        group_start_bit_pos,
        .{ 280, 256, 256, 256, 40 },
        2,
        1,
        0,
        8,
    );
    try testing.expectEqual(event_stream_start_bit_pos, stream.event_stream_start_bit_pos);
    try testing.expectEqual(bit_pos, stream.end_bit_pos);
    try testing.expectEqual(@as(usize, 2), stream.event_count);
    try testing.expectEqual(@as(usize, 2), stream.emitted_pixels);
    try testing.expectEqual(@as(usize, 2), stream.preview_len);
    try testing.expectEqual(imaging.Vp8lEventKind.literal, stream.preview[0].kind);
    try testing.expectEqual(@as(u16, 1), stream.preview[0].green);
    try testing.expectEqual(@as(u16, 10), stream.preview[0].red);
    try testing.expectEqual(@as(u16, 20), stream.preview[0].blue);
    try testing.expectEqual(@as(u16, 255), stream.preview[0].alpha);
    try testing.expectEqual(imaging.Vp8lEventKind.literal, stream.preview[1].kind);
    try testing.expectEqual(@as(u16, 2), stream.preview[1].green);
    try testing.expectEqual(@as(u16, 10), stream.preview[1].red);
    try testing.expectEqual(@as(u16, 20), stream.preview[1].blue);
    try testing.expectEqual(@as(u16, 255), stream.preview[1].alpha);
}
