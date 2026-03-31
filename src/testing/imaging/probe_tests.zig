const std = @import("std");
const imaging = @import("Pixio");
const helpers = @import("helpers.zig");

test "probeInfo reads repository sample png metadata" {
    const testing = std.testing;

    const info = try imaging.probeFileInfo(testing.allocator, "data/archive/images/000_0001.png");
    try testing.expectEqual(imaging.ImageFormat.png, info.format);
    try testing.expectEqual(@as(usize, 134), info.width);
    try testing.expectEqual(@as(usize, 128), info.height);
    try testing.expectEqual(@as(usize, 3), info.channels);
    try testing.expect(!info.has_alpha);
}

test "probeInfo reads lossless webp metadata" {
    const testing = std.testing;

    const webp = try helpers.decodeBase64Alloc(testing.allocator, "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=");
    defer testing.allocator.free(webp);

    const info = try imaging.probeInfo(webp);
    try testing.expectEqual(imaging.ImageFormat.webp, info.format);
    try testing.expectEqual(@as(usize, 2), info.width);
    try testing.expectEqual(@as(usize, 1), info.height);
    try testing.expectEqual(@as(usize, 3), info.channels);
    try testing.expect(!info.has_alpha);
}

test "probeWebpInfo distinguishes lossless and lossy bitstreams" {
    const testing = std.testing;

    const lossless = try helpers.decodeBase64Alloc(testing.allocator, "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=");
    defer testing.allocator.free(lossless);

    const lossy = try helpers.decodeBase64Alloc(testing.allocator, "UklGRkgAAABXRUJQVlA4IDwAAAAwAgCdASoCAAEAAAAAJaACdLoB+AADIQb7gAD5f/8uv//vTP/5zIj//2Z7/Znv9me/+zPf/maJjmP16AA=");
    defer testing.allocator.free(lossy);

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

test "probeWebpInfo reports alpha for lossless rgba sample" {
    const testing = std.testing;

    const webp = try helpers.decodeBase64Alloc(testing.allocator, "UklGRhwAAABXRUJQVlA4TA8AAAAvAAAAEAcQ/Y8CBiKi/wEA");
    defer testing.allocator.free(webp);

    const info = try imaging.probeWebpInfo(webp);
    try testing.expectEqual(@as(usize, 1), info.width);
    try testing.expectEqual(@as(usize, 1), info.height);
    try testing.expectEqual(@as(imaging.WebpInfo, info).kind, .vp8l);
    try testing.expect(info.has_alpha);
}

test "findPrimaryChunk identifies webp payload kind" {
    const testing = std.testing;

    const bytes = try helpers.decodeBase64Alloc(testing.allocator, "UklGRh4AAABXRUJQVlA4TBEAAAAvAQAAAA+w//Mf8x8VMqL/AQA=");
    defer testing.allocator.free(bytes);

    const tag = try imaging.probeWebpPrimaryChunkTag(bytes);
    try testing.expectEqual(imaging.WebpChunkTag.vp8l, tag);
}
