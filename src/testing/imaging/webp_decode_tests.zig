const std = @import("std");
const imaging = @import("Pixio");
const helpers = @import("helpers.zig");

test "decodeVp8lSingleGroupArgbAtBitPos decodes literal-only pixels" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 16;
    var bit_pos: usize = 0;

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 1, 8);
    helpers.writeBits(&payload, &bit_pos, 2, 8);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 10, 8);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 20, 8);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 0, 8);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    var image = try imaging.decodeVp8lSingleGroupArgbAtBitPos(
        testing.allocator,
        &payload,
        0,
        .{ 280, 256, 256, 256, 40 },
        2,
        1,
        0,
    );
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(bit_pos, image.end_bit_pos);
    try testing.expectEqual(@as(usize, 2), image.pixels.len);
    try testing.expectEqual(@as(u32, 0xff0a0114), image.pixels[0]);
    try testing.expectEqual(@as(u32, 0xff0a0214), image.pixels[1]);
}

test "decodeRgb8 decodes synthetic single-group lossless webp" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 32;
    payload[0] = 0x2f;
    helpers.writeVp8lHeader(payload[1..5], 2, 1, false);

    var bit_pos: usize = 40;
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 1, 8);
    helpers.writeBits(&payload, &bit_pos, 2, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 10, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 20, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 0, 8);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    const payload_size = (bit_pos + 7) / 8;
    const riff_size = 4 + 8 + payload_size;
    const webp_size = 12 + 8 + payload_size;
    const webp = try testing.allocator.alloc(u8, webp_size);
    defer testing.allocator.free(webp);
    @memcpy(webp[0..4], "RIFF");
    helpers.writeU32le(webp[4..8], @intCast(riff_size));
    @memcpy(webp[8..12], "WEBP");
    @memcpy(webp[12..16], "VP8L");
    helpers.writeU32le(webp[16..20], @intCast(payload_size));
    @memcpy(webp[20 .. 20 + payload_size], payload[0..payload_size]);

    var image = try imaging.decodeRgb8(testing.allocator, webp);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 1, 20 }, image.data[0..3]);
    try testing.expectEqualSlices(u8, &[_]u8{ 10, 2, 20 }, image.data[3..6]);
}

test "decodeVp8lPayloadArgb decodes color-indexing single-transform payload" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 40;
    payload[0] = 0x2f;
    helpers.writeVp8lHeader(payload[1..5], 2, 1, false);

    var bit_pos: usize = 40;
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 3, 2);
    helpers.writeBits(&payload, &bit_pos, 1, 8);

    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 0, 8);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    const payload_size = (bit_pos + 7) / 8;
    var image = try imaging.decodeVp8lPayloadArgb(testing.allocator, payload[0..payload_size]);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(bit_pos, image.end_bit_pos);
    try testing.expectEqual(@as(u32, 0xff000000), image.pixels[0]);
    try testing.expectEqual(@as(u32, 0xfe00ff00), image.pixels[1]);
}

test "decodeRgb8 decodes synthetic color-indexing lossless webp" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 40;
    payload[0] = 0x2f;
    helpers.writeVp8lHeader(payload[1..5], 2, 1, false);

    var bit_pos: usize = 40;
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 3, 2);
    helpers.writeBits(&payload, &bit_pos, 1, 8);

    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 0, 8);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    const payload_size = (bit_pos + 7) / 8;
    const riff_size = 4 + 8 + payload_size;
    const webp_size = 12 + 8 + payload_size;
    const webp = try testing.allocator.alloc(u8, webp_size);
    defer testing.allocator.free(webp);
    @memcpy(webp[0..4], "RIFF");
    helpers.writeU32le(webp[4..8], @intCast(riff_size));
    @memcpy(webp[8..12], "WEBP");
    @memcpy(webp[12..16], "VP8L");
    helpers.writeU32le(webp[16..20], @intCast(payload_size));
    @memcpy(webp[20 .. 20 + payload_size], payload[0..payload_size]);

    var image = try imaging.decodeRgb8(testing.allocator, webp);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0 }, image.data[0..3]);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 255, 0 }, image.data[3..6]);
}

test "decodeVp8lPayloadArgb decodes subtract-green single-group payload" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 32;
    payload[0] = 0x2f;
    helpers.writeVp8lHeader(payload[1..5], 2, 1, false);

    var bit_pos: usize = 40;
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 2);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 5, 8);
    helpers.writeBits(&payload, &bit_pos, 9, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 7, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 1, 8);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    const payload_size = (bit_pos + 7) / 8;
    var image = try imaging.decodeVp8lPayloadArgb(testing.allocator, payload[0..payload_size]);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(bit_pos, image.end_bit_pos);
    try testing.expectEqual(@as(usize, 2), image.pixels.len);
    try testing.expectEqual(@as(u32, 0xff07050c), image.pixels[0]);
    try testing.expectEqual(@as(u32, 0xff0b0910), image.pixels[1]);
}

test "decodeRgb8 decodes synthetic subtract-green lossless webp" {
    const testing = std.testing;

    var payload = [_]u8{0} ** 32;
    payload[0] = 0x2f;
    helpers.writeVp8lHeader(payload[1..5], 2, 1, false);

    var bit_pos: usize = 40;
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 2);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 0);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 5, 8);
    helpers.writeBits(&payload, &bit_pos, 9, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 2, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 7, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 255, 8);

    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);
    helpers.writeBits(&payload, &bit_pos, 1, 8);

    helpers.writeBit(&payload, &bit_pos, 0);
    helpers.writeBit(&payload, &bit_pos, 1);

    const payload_size = (bit_pos + 7) / 8;
    const riff_size = 4 + 8 + payload_size;
    const webp_size = 12 + 8 + payload_size;
    const webp = try testing.allocator.alloc(u8, webp_size);
    defer testing.allocator.free(webp);
    @memcpy(webp[0..4], "RIFF");
    helpers.writeU32le(webp[4..8], @intCast(riff_size));
    @memcpy(webp[8..12], "WEBP");
    @memcpy(webp[12..16], "VP8L");
    helpers.writeU32le(webp[16..20], @intCast(payload_size));
    @memcpy(webp[20 .. 20 + payload_size], payload[0..payload_size]);

    var image = try imaging.decodeRgb8(testing.allocator, webp);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqualSlices(u8, &[_]u8{ 7, 5, 12 }, image.data[0..3]);
    try testing.expectEqualSlices(u8, &[_]u8{ 11, 9, 16 }, image.data[3..6]);
}

test "decodeRgb8 decodes real color-indexed solid red webp" {
    const testing = std.testing;

    const webp = try helpers.decodeBase64Alloc(testing.allocator, "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAAAcQ/Y/+ByKi/wEA");
    defer testing.allocator.free(webp);

    var image = try imaging.decodeRgb8(testing.allocator, webp);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 1), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqualSlices(u8, &[_]u8{ 255, 0, 0 }, image.data[0..3]);
}
