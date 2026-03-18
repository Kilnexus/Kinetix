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
    try testing.expectEqual(imaging.ImageFormat.gif, imaging.detectFormat("GIF89arest"));
    try testing.expectEqual(imaging.ImageFormat.ico, imaging.detectFormat("\x00\x00\x01\x00rest"));
    try testing.expectEqual(imaging.ImageFormat.webp, imaging.detectFormat("RIFF\x1a\x00\x00\x00WEBP"));
}

test "probeInfo reads repository sample png metadata" {
    const testing = @import("std").testing;

    const info = try imaging.probeFileInfo(testing.allocator, "data/archive/images/000_0001.png");
    try testing.expectEqual(imaging.ImageFormat.png, info.format);
    try testing.expectEqual(@as(usize, 134), info.width);
    try testing.expectEqual(@as(usize, 128), info.height);
    try testing.expectEqual(@as(usize, 3), info.channels);
    try testing.expect(!info.has_alpha);
}

test "probeInfo reads lossless webp metadata" {
    const std = @import("std");
    const testing = std.testing;

    const webp_base64 = "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=";
    const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(webp_base64);
    const webp = try testing.allocator.alloc(u8, decoded_len);
    defer testing.allocator.free(webp);
    try std.base64.standard.Decoder.decode(webp, webp_base64);

    const info = try imaging.probeInfo(webp);
    try testing.expectEqual(imaging.ImageFormat.webp, info.format);
    try testing.expectEqual(@as(usize, 2), info.width);
    try testing.expectEqual(@as(usize, 1), info.height);
    try testing.expectEqual(@as(usize, 3), info.channels);
    try testing.expect(!info.has_alpha);
}

test "probeWebpInfo distinguishes lossless and lossy bitstreams" {
    const std = @import("std");
    const testing = std.testing;

    const lossless_base64 = "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=";
    const lossy_base64 = "UklGRkgAAABXRUJQVlA4IDwAAAAwAgCdASoCAAEAAAAAJaACdLoB+AADIQb7gAD5f/8uv//vTP/5zIj//2Z7/Znv9me/+zPf/maJjmP16AA=";

    const lossless_len = try std.base64.standard.Decoder.calcSizeForSlice(lossless_base64);
    const lossless = try testing.allocator.alloc(u8, lossless_len);
    defer testing.allocator.free(lossless);
    try std.base64.standard.Decoder.decode(lossless, lossless_base64);

    const lossy_len = try std.base64.standard.Decoder.calcSizeForSlice(lossy_base64);
    const lossy = try testing.allocator.alloc(u8, lossy_len);
    defer testing.allocator.free(lossy);
    try std.base64.standard.Decoder.decode(lossy, lossy_base64);

    const lossless_info = try imaging.probeWebpInfo(lossless);
    try testing.expectEqual(@as(usize, 2), lossless_info.width);
    try testing.expectEqual(@as(usize, 1), lossless_info.height);
    try testing.expectEqual(@as(imaging.WebpInfo, lossless_info).kind, .vp8l);
    try testing.expect(!lossless_info.has_alpha);
    try testing.expect(!lossless_info.is_animated);

    const lossy_info = try imaging.probeWebpInfo(lossy);
    try testing.expectEqual(@as(usize, 2), lossy_info.width);
    try testing.expectEqual(@as(usize, 1), lossy_info.height);
    try testing.expectEqual(@as(imaging.WebpInfo, lossy_info).kind, .vp8);
    try testing.expect(!lossy_info.has_alpha);
    try testing.expect(!lossy_info.is_animated);
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

test "decodeRgb8 decodes palette gif" {
    const std = @import("std");
    const testing = std.testing;

    const gif_base64 =
        "R0lGODlhAgABAPcAAAAAAAAAMwAAZgAAmQAAzAAA/wArAAArMwArZgArmQArzAAr/wBVAABVMwBVZgBVmQBVzABV"
        ++ "/wCAAACAMwCAZgCAmQCAzACA/wCqAACqMwCqZgCqmQCqzACq/wDVAADVMwDVZgDVmQDVzADV/wD/AAD/MwD/ZgD/"
        ++ "mQD/zAD//zMAADMAMzMAZjMAmTMAzDMA/zMrADMrMzMrZjMrmTMrzDMr/zNVADNVMzNVZjNVmTNVzDNV/zOAADOA"
        ++ "MzOAZjOAmTOAzDOA/zOqADOqMzOqZjOqmTOqzDOq/zPVADPVMzPVZjPVmTPVzDPV/zP/ADP/MzP/ZjP/mTP/zDP/"
        ++ "/2YAAGYAM2YAZmYAmWYAzGYA/2YrAGYrM2YrZmYrmWYrzGYr/2ZVAGZVM2ZVZmZVmWZVzGZV/2aAAGaAM2aAZmaA"
        ++ "mWaAzGaA/2aqAGaqM2aqZmaqmWaqzGaq/2bVAGbVM2bVZmbVmWbVzGbV/2b/AGb/M2b/Zmb/mWb/zGb//5kAAJkA"
        ++ "M5kAZpkAmZkAzJkA/5krAJkrM5krZpkrmZkrzJkr/5lVAJlVM5lVZplVmZlVzJlV/5mAAJmAM5mAZpmAmZmAzJmA"
        ++ "/5mqAJmqM5mqZpmqmZmqzJmq/5nVAJnVM5nVZpnVmZnVzJnV/5n/AJn/M5n/Zpn/mZn/zJn//8wAAMwAM8wAZswA"
        ++ "mcwAzMwA/8wrAMwrM8wrZswrmcwrzMwr/8xVAMxVM8xVZsxVmcxVzMxV/8yAAMyAM8yAZsyAmcyAzMyA/8yqAMyq"
        ++ "M8yqZsyqmcyqzMyq/8zVAMzVM8zVZszVmczVzMzV/8z/AMz/M8z/Zsz/mcz/zMz///8AAP8AM/8AZv8Amf8AzP8A"
        ++ "//8rAP8rM/8rZv8rmf8rzP8r//9VAP9VM/9VZv9Vmf9VzP9V//+AAP+AM/+AZv+Amf+AzP+A//+qAP+qM/+qZv+q"
        ++ "mf+qzP+q///VAP/VM//VZv/Vmf/VzP/V////AP//M///Zv//mf//zP///wAAAAAAAAAAAAAAACH5BAEAAPwALAAA"
        ++ "AAACAAEAAAgFAKWRCAgAOw==";

    const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(gif_base64);
    const gif = try testing.allocator.alloc(u8, decoded_len);
    defer testing.allocator.free(gif);
    try std.base64.standard.Decoder.decode(gif, gif_base64);

    var image = try imaging.decodeRgb8(testing.allocator, gif);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 2), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
    try testing.expect(image.data[0] > image.data[1]);
    try testing.expect(image.data[0] > image.data[2]);
    try testing.expect(image.data[4] > image.data[3]);
    try testing.expect(image.data[4] > image.data[5]);
}

test "decodeRgb8 decodes png-backed ico" {
    const std = @import("std");
    const testing = std.testing;

    const png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAANSURBVBhXY/jPwPAfAAUAAf+mXJtdAAAAAElFTkSuQmCC";
    const png_len = try std.base64.standard.Decoder.calcSizeForSlice(png_base64);
    const png_bytes = try testing.allocator.alloc(u8, png_len);
    defer testing.allocator.free(png_bytes);
    try std.base64.standard.Decoder.decode(png_bytes, png_base64);

    const ico_len = 6 + 16 + png_len;
    const ico = try testing.allocator.alloc(u8, ico_len);
    defer testing.allocator.free(ico);

    ico[0] = 0x00;
    ico[1] = 0x00;
    ico[2] = 0x01;
    ico[3] = 0x00;
    ico[4] = 0x01;
    ico[5] = 0x00;

    ico[6] = 0x01;
    ico[7] = 0x01;
    ico[8] = 0x00;
    ico[9] = 0x00;
    ico[10] = 0x01;
    ico[11] = 0x00;
    ico[12] = 0x20;
    ico[13] = 0x00;

    writeU32le(ico[14..18], @intCast(png_len));
    writeU32le(ico[18..22], 22);
    @memcpy(ico[22..], png_bytes);

    var image = try imaging.decodeRgb8(testing.allocator, ico);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 1), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
    try testing.expect(image.data[0] > image.data[1]);
    try testing.expect(image.data[0] > image.data[2]);
}

test "decodeRgb8 decodes bmp-backed ico" {
    const testing = @import("std").testing;

    const payload_len: usize = 40 + 4 + 4;
    const ico_len: usize = 6 + 16 + payload_len;
    const ico = try testing.allocator.alloc(u8, ico_len);
    defer testing.allocator.free(ico);
    @memset(ico, 0);

    ico[0] = 0x00;
    ico[1] = 0x00;
    ico[2] = 0x01;
    ico[3] = 0x00;
    ico[4] = 0x01;
    ico[5] = 0x00;

    ico[6] = 0x01;
    ico[7] = 0x01;
    ico[8] = 0x00;
    ico[9] = 0x00;
    ico[10] = 0x01;
    ico[11] = 0x00;
    ico[12] = 0x20;
    ico[13] = 0x00;
    writeU32le(ico[14..18], payload_len);
    writeU32le(ico[18..22], 22);

    const dib = ico[22..];
    writeU32le(dib[0..4], 40);
    writeU32le(dib[4..8], 1);
    writeU32le(dib[8..12], 2);
    dib[12] = 0x01;
    dib[13] = 0x00;
    dib[14] = 0x20;
    dib[15] = 0x00;
    writeU32le(dib[16..20], 0);
    writeU32le(dib[20..24], 4);
    writeU32le(dib[24..28], 0);
    writeU32le(dib[28..32], 0);
    writeU32le(dib[32..36], 0);
    writeU32le(dib[36..40], 0);

    dib[40] = 0x00;
    dib[41] = 0x00;
    dib[42] = 0xff;
    dib[43] = 0xff;
    dib[44] = 0x00;
    dib[45] = 0x00;
    dib[46] = 0x00;
    dib[47] = 0x00;

    var image = try imaging.decodeRgb8(testing.allocator, ico);
    defer image.deinit();

    try testing.expectEqual(@as(usize, 1), image.width);
    try testing.expectEqual(@as(usize, 1), image.height);
    try testing.expectEqual(@as(usize, 3), image.channels);
    try testing.expect(image.data[0] > image.data[1]);
    try testing.expect(image.data[0] > image.data[2]);
}

fn writeU32le(dst: []u8, value: u32) void {
    dst[0] = @intCast(value & 0xff);
    dst[1] = @intCast((value >> 8) & 0xff);
    dst[2] = @intCast((value >> 16) & 0xff);
    dst[3] = @intCast((value >> 24) & 0xff);
}
