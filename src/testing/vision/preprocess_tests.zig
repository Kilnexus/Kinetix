const std = @import("std");
const imaging = @import("Pixio");
const runtime = @import("runtime");
const preprocess = @import("vision");

test "prepareImageAsTensor matches reference letterbox path" {
    const testing = std.testing;

    var src = try imaging.ImageU8.init(testing.allocator, 3, 2, 3);
    defer src.deinit();
    const pixels = [_]u8{
        10, 20, 30,
        40, 50, 60,
        70, 80, 90,
        15, 25, 35,
        45, 55, 65,
        75, 85, 95,
    };
    @memcpy(src.data, &pixels);

    var prepared = try preprocess.prepareImageAsTensor(testing.allocator, &src, 6);
    defer prepared.deinit();

    var boxed = try imaging.letterboxImage(testing.allocator, &src, 6, 6, 114);
    defer boxed.deinit();

    var reference = try runtime.Tensor.init(testing.allocator, 1, 3, boxed.image.height, boxed.image.width);
    defer reference.deinit();
    for (0..boxed.image.height) |y| {
        for (0..boxed.image.width) |x| {
            const dst_index = y * boxed.image.width + x;
            for (0..3) |channel| {
                reference.data[channel * boxed.image.width * boxed.image.height + dst_index] =
                    @as(f32, @floatFromInt(boxed.image.get(x, y, channel))) / 255.0;
            }
        }
    }

    try testing.expectEqual(boxed.info.src_width, prepared.info.src_width);
    try testing.expectEqual(boxed.info.src_height, prepared.info.src_height);
    try testing.expectEqual(boxed.info.resized_width, prepared.info.resized_width);
    try testing.expectEqual(boxed.info.resized_height, prepared.info.resized_height);
    try testing.expectEqual(boxed.info.pad_left, prepared.info.pad_left);
    try testing.expectEqual(boxed.info.pad_top, prepared.info.pad_top);
    try testing.expectEqual(reference.shape, prepared.tensor.shape);

    for (reference.data, prepared.tensor.data) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
}
