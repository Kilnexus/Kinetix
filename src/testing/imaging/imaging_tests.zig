const imaging = @import("imaging");

test "resizeBilinear preserves shape metadata" {
    const testing = @import("std").testing;

    var src = try imaging.ImageU8.init(testing.allocator, 4, 3, 3);
    defer src.deinit();
    src.fill(10);

    var dst = try imaging.resizeBilinear(testing.allocator, &src, 8, 6);
    defer dst.deinit();

    try testing.expectEqual(@as(usize, 8), dst.width);
    try testing.expectEqual(@as(usize, 6), dst.height);
    try testing.expectEqual(@as(usize, 3), dst.channels);
}

test "letterboxImage computes centered padding" {
    const testing = @import("std").testing;

    var src = try imaging.ImageU8.init(testing.allocator, 100, 50, 3);
    defer src.deinit();
    src.fill(255);

    var boxed = try imaging.letterboxImage(testing.allocator, &src, 160, 160, 114);
    defer boxed.deinit();

    try testing.expectEqual(@as(usize, 160), boxed.image.width);
    try testing.expectEqual(@as(usize, 160), boxed.image.height);
    try testing.expectEqual(@as(usize, 160), boxed.info.resized_width);
    try testing.expectEqual(@as(usize, 80), boxed.info.resized_height);
    try testing.expectEqual(@as(usize, 0), boxed.info.pad_left);
    try testing.expectEqual(@as(usize, 40), boxed.info.pad_top);
}

test "detectFormat recognizes png and bmp signatures" {
    const testing = @import("std").testing;

    try testing.expectEqual(imaging.ImageFormat.png, imaging.detectFormat("\x89PNG\r\n\x1a\nrest"));
    try testing.expectEqual(imaging.ImageFormat.bmp, imaging.detectFormat("BMrest"));
    try testing.expectEqual(imaging.ImageFormat.jpeg, imaging.detectFormat("\xff\xd8\xff\xe0"));
}

test "decodeRgb8 decodes repository sample png natively" {
    const testing = @import("std").testing;

    var image = try imaging.decodeFileRgb8(testing.allocator, "data/archive/images/000_0001.png");
    defer image.deinit();

    try testing.expectEqual(@as(usize, 134), image.width);
    try testing.expectEqual(@as(usize, 128), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
}

test "decodeRgb8 decodes 24-bit bmp" {
    const testing = @import("std").testing;

    const bmp = [_]u8{
        0x42, 0x4d, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
        0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00,
        0x13, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0x00,
    };

    var image = try imaging.decodeRgb8(testing.allocator, &bmp);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqualSlices(u8, &[_]u8{ 0xff, 0x00, 0x00 }, image.data[0..3]);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x00, 0xff, 0x00 }, image.data[3..6]);
}

test "decodeRgb8 decodes baseline jpeg" {
    const std = @import("std");
    const testing = std.testing;

    const jpeg_base64 =
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIs"
        ++ "IxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
        ++ "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAA"
        ++ "AAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAk"
        ++ "M2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKT"
        ++ "lJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QA"
        ++ "HwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdh"
        ++ "cRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hp"
        ++ "anN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk"
        ++ "5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwC7p3/IMtP+uKf+giiiivzefxM/Lcy/32t/il+bP//Z";

    const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(jpeg_base64);
    const jpeg = try testing.allocator.alloc(u8, decoded_len);
    defer testing.allocator.free(jpeg);
    try std.base64.standard.Decoder.decode(jpeg, jpeg_base64);

    var image = try imaging.decodeRgb8(testing.allocator, jpeg);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
    try testing.expect(image.data[2] < image.data[0]);
    try testing.expect(image.data[2] < image.data[1]);
    try testing.expect(image.data[4] > image.data[3]);
    try testing.expect(image.data[4] > image.data[5]);
    try testing.expect(!std.mem.eql(u8, image.data[0..3], image.data[3..6]));
}
