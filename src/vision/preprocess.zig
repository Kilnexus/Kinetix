const std = @import("std");
const imaging = @import("imaging");
const runtime = @import("runtime");
const image = @import("image.zig");

pub const LetterboxInfo = imaging.LetterboxInfo;

pub const PreparedInput = struct {
    tensor: runtime.Tensor,
    info: LetterboxInfo,

    pub fn deinit(self: *PreparedInput) void {
        self.tensor.deinit();
        self.* = undefined;
    }
};

pub fn loadImageAsTensor(
    allocator: std.mem.Allocator,
    path: []const u8,
    input_size: usize,
) !PreparedInput {
    var src = try image.loadRgb8(allocator, path);
    defer src.deinit();

    var boxed = try imaging.letterboxImage(allocator, &src, input_size, input_size, 114);
    defer boxed.deinit();

    var tensor = try runtime.Tensor.init(allocator, 1, 3, boxed.image.height, boxed.image.width);
    errdefer tensor.deinit();
    for (0..boxed.image.height) |y| {
        for (0..boxed.image.width) |x| {
            for (0..3) |channel| {
                const value = @as(f32, @floatFromInt(boxed.image.get(x, y, channel))) / 255.0;
                tensor.set(0, channel, y, x, value);
            }
        }
    }

    return .{
        .tensor = tensor,
        .info = .{
            .src_width = boxed.info.src_width,
            .src_height = boxed.info.src_height,
            .dst_width = boxed.info.dst_width,
            .dst_height = boxed.info.dst_height,
            .resized_width = boxed.info.resized_width,
            .resized_height = boxed.info.resized_height,
            .pad_left = boxed.info.pad_left,
            .pad_top = boxed.info.pad_top,
            .scale_x = boxed.info.scale_x,
            .scale_y = boxed.info.scale_y,
        },
    };
}

pub fn remapDetectionsToSource(detections: []runtime.Detection, info: LetterboxInfo) void {
    for (detections) |*det| {
        var box = imaging.BoxF32{ .x1 = det.x1, .y1 = det.y1, .x2 = det.x2, .y2 = det.y2 };
        imaging.remapLetterboxedBoxToSource(
            &box,
            info.pad_left,
            info.pad_top,
            info.scale_x,
            info.scale_y,
            info.src_width,
            info.src_height,
        );
        det.x1 = box.x1;
        det.y1 = box.y1;
        det.x2 = box.x2;
        det.y2 = box.y2;
    }
}
