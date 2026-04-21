const std = @import("std");
const imaging = @import("Pixio");
const runtime = @import("runtime");
const image = @import("image.zig");

pub const LetterboxInfo = imaging.LetterboxInfo;
const pad_value_u8: u8 = 114;

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

    return prepareImageAsTensor(allocator, &src, input_size);
}

pub fn prepareImageAsTensor(
    allocator: std.mem.Allocator,
    src: *const imaging.ImageU8,
    input_size: usize,
) !PreparedInput {
    if (src.channels < 3) return error.InvalidChannelCount;

    var prepared = try imaging.prepareTensor(allocator, src, .{
        .target_width = input_size,
        .target_height = input_size,
        .mode = .letterbox,
        .kernel = .bilinear,
        .output_pixel_format = .rgb8,
        .pad_value = pad_value_u8,
        .normalize = .{},
    });
    defer prepared.deinit();

    var tensor = try runtime.Tensor.init(allocator, 1, prepared.tensor.channels, prepared.tensor.height, prepared.tensor.width);
    errdefer tensor.deinit();
    @memcpy(tensor.data, prepared.tensor.data);

    const info = LetterboxInfo{
        .src_width = prepared.info.src_width,
        .src_height = prepared.info.src_height,
        .dst_width = prepared.info.output_width,
        .dst_height = prepared.info.output_height,
        .resized_width = prepared.info.resized_width,
        .resized_height = prepared.info.resized_height,
        .pad_left = prepared.info.offset_x,
        .pad_top = prepared.info.offset_y,
        .scale_x = prepared.info.scale_x,
        .scale_y = prepared.info.scale_y,
    };

    return .{
        .tensor = tensor,
        .info = info,
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

