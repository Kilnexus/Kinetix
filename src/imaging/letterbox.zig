const std = @import("std");
const resize = @import("resize.zig");
const types = @import("types.zig");

pub const ImageU8 = types.ImageU8;

pub const LetterboxInfo = struct {
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    resized_width: usize,
    resized_height: usize,
    pad_left: usize,
    pad_top: usize,
    scale_x: f32,
    scale_y: f32,
};

pub const LetterboxedImage = struct {
    image: ImageU8,
    info: LetterboxInfo,

    pub fn deinit(self: *LetterboxedImage) void {
        self.image.deinit();
        self.* = undefined;
    }
};

pub fn letterbox(
    allocator: std.mem.Allocator,
    src: *const ImageU8,
    dst_width: usize,
    dst_height: usize,
    pad_value: u8,
) !LetterboxedImage {
    const scale = @min(
        @as(f32, @floatFromInt(dst_width)) / @as(f32, @floatFromInt(src.width)),
        @as(f32, @floatFromInt(dst_height)) / @as(f32, @floatFromInt(src.height)),
    );
    const resized_width = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src.width)) * scale))));
    const resized_height = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src.height)) * scale))));
    const pad_left = (dst_width - resized_width) / 2;
    const pad_top = (dst_height - resized_height) / 2;

    var resized = try resize.resizeBilinear(allocator, src, resized_width, resized_height);
    defer resized.deinit();

    var canvas = try ImageU8.init(allocator, dst_width, dst_height, src.channels);
    errdefer canvas.deinit();
    canvas.fill(pad_value);

    for (0..resized_height) |y| {
        for (0..resized_width) |x| {
            for (0..src.channels) |channel| {
                canvas.set(pad_left + x, pad_top + y, channel, resized.get(x, y, channel));
            }
        }
    }

    return .{
        .image = canvas,
        .info = .{
            .src_width = src.width,
            .src_height = src.height,
            .dst_width = dst_width,
            .dst_height = dst_height,
            .resized_width = resized_width,
            .resized_height = resized_height,
            .pad_left = pad_left,
            .pad_top = pad_top,
            .scale_x = @as(f32, @floatFromInt(resized_width)) / @as(f32, @floatFromInt(src.width)),
            .scale_y = @as(f32, @floatFromInt(resized_height)) / @as(f32, @floatFromInt(src.height)),
        },
    };
}
