const std = @import("std");
const imaging = @import("Pixio");

test "resizeBilinear preserves shape metadata" {
    const testing = std.testing;

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
    const testing = std.testing;

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
    const testing = std.testing;

    try testing.expectEqual(imaging.ImageFormat.png, imaging.detectFormat("\x89PNG\r\n\x1a\nrest"));
    try testing.expectEqual(imaging.ImageFormat.bmp, imaging.detectFormat("BMrest"));
    try testing.expectEqual(imaging.ImageFormat.jpeg, imaging.detectFormat("\xff\xd8\xff\xe0"));
    try testing.expectEqual(imaging.ImageFormat.gif, imaging.detectFormat("GIF89arest"));
    try testing.expectEqual(imaging.ImageFormat.ico, imaging.detectFormat("\x00\x00\x01\x00rest"));
    try testing.expectEqual(imaging.ImageFormat.webp, imaging.detectFormat("RIFF\x1a\x00\x00\x00WEBP"));
}
