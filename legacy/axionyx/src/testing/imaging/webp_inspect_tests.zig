const std = @import("std");
const imaging = @import("Pixio");
const helpers = @import("helpers.zig");

const writeBit = helpers.writeBit;
const writeBits = helpers.writeBits;

test "inspectWebpVp8l parses transform chain for lossless samples" {
    const testing = std.testing;

    const rgb = try helpers.decodeBase64Alloc(testing.allocator, "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=");
    defer testing.allocator.free(rgb);
    const rgba = try helpers.decodeBase64Alloc(testing.allocator, "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAEAcQ/Y8CBiKi/wEA");
    defer testing.allocator.free(rgba);

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
    const testing = std.testing;

    const samples = [_]struct {
        name: []const u8,
        base64: []const u8,
    }{
        .{ .name = "solid_red_1x1", .base64 = "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAAAcQ/Y/+ByKi/wEA" },
        .{ .name = "checker_4x4", .base64 = "UklGRiYAAABXRUJQVlA4TBoAAAAvA8AAAA8w//M///MfeFDTtgGLr6Qjov/BOQ==" },
        .{ .name = "gradient_16x16", .base64 = "UklGRrAAAABXRUJQVlA4TKMAAAAvD8ADEE1kRP9jEYUf8P5HAUHbtjGE8Ke7q6cwEIwhSRJ0GIUyKIuyKItyKIOSTwFJ0vPwuSK3bZtjdtln8LEts6NYObSTOI+5yLhbHrab8Wy5bE/G3QpaiXcl2pto5aWVxJaxfRnb8rEt49sydiTOyp92Ev+VQ/snzirYbsbdco25WIneUWR+8PBRO0l8d6c9urvTHu7utMd3d9qjrX7+7QMXAA==" },
    };

    for (samples) |sample| {
        const webp = try helpers.decodeBase64Alloc(testing.allocator, sample.base64);
        defer testing.allocator.free(webp);

        const info = try imaging.inspectWebpVp8l(webp);
        _ = sample.name;
        try testing.expect(!info.tail_flags_known);
        try testing.expectEqual(@as(?usize, null), info.image_data_start_bit_pos);
        try testing.expectEqual(@as(?imaging.Vp8lImageDataHeader, null), info.main_image_header);
    }
}

test "inspectVp8lImageDataAtBitPos parses argb meta prefix branch" {
    const testing = std.testing;

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
    const testing = std.testing;

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
