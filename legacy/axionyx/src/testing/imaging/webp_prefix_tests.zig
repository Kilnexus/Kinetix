const std = @import("std");
const imaging = @import("Pixio");
const helpers = @import("helpers.zig");

const writeBit = helpers.writeBit;
const writeBits = helpers.writeBits;

test "inspectVp8lNormalPrefixCodeAtBitPos decodes literal code length sequence" {
    const testing = std.testing;

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
    const testing = std.testing;

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
    const testing = std.testing;

    const code_lengths = [_]u8{ 0, 0, 1, 0 };
    const payload = [_]u8{ 0xaa, 0x55 };

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
    const testing = std.testing;

    const code_lengths = [_]u8{ 3, 3, 3, 3, 3, 0, 0, 0, 0, 0 };
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
    const testing = std.testing;

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
    const testing = std.testing;

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
    const testing = std.testing;

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

    const event_stream_start_bit_pos = bit_pos;
    writeBit(&payload, &bit_pos, 0);
    writeBit(&payload, &bit_pos, 1);

    const stream = try imaging.inspectVp8lEventStreamAtBitPos(
        &payload,
        0,
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
