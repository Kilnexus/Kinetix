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

    const info = computeLetterboxInfo(src.width, src.height, input_size, input_size);
    var tensor = try runtime.Tensor.init(allocator, 1, 3, input_size, input_size);
    errdefer tensor.deinit();

    const pad_value = @as(f32, @floatFromInt(pad_value_u8)) / 255.0;
    tensor.fill(pad_value);

    const plane = input_size * input_size;
    for (0..info.resized_height) |dy| {
        const src_y = ((@as(f32, @floatFromInt(dy)) + 0.5) * @as(f32, @floatFromInt(src.height)) / @as(f32, @floatFromInt(info.resized_height))) - 0.5;
        const y0 = clampIndex(@as(isize, @intFromFloat(@floor(src_y))), src.height);
        const y1 = if (y0 + 1 < src.height) y0 + 1 else src.height - 1;
        const wy = src_y - @as(f32, @floatFromInt(y0));
        const dst_y = info.pad_top + dy;

        for (0..info.resized_width) |dx| {
            const src_x = ((@as(f32, @floatFromInt(dx)) + 0.5) * @as(f32, @floatFromInt(src.width)) / @as(f32, @floatFromInt(info.resized_width))) - 0.5;
            const x0 = clampIndex(@as(isize, @intFromFloat(@floor(src_x))), src.width);
            const x1 = if (x0 + 1 < src.width) x0 + 1 else src.width - 1;
            const wx = src_x - @as(f32, @floatFromInt(x0));
            const dst_x = info.pad_left + dx;
            const dst_index = dst_y * input_size + dst_x;

            const p00_index = (y0 * src.width + x0) * src.channels;
            const p10_index = (y0 * src.width + x1) * src.channels;
            const p01_index = (y1 * src.width + x0) * src.channels;
            const p11_index = (y1 * src.width + x1) * src.channels;

            for (0..3) |channel| {
                const p00 = @as(f32, @floatFromInt(src.data[p00_index + channel]));
                const p10 = @as(f32, @floatFromInt(src.data[p10_index + channel]));
                const p01 = @as(f32, @floatFromInt(src.data[p01_index + channel]));
                const p11 = @as(f32, @floatFromInt(src.data[p11_index + channel]));
                const top = lerp(p00, p10, wx);
                const bottom = lerp(p01, p11, wx);
                const sampled: u8 = @intFromFloat(@round(lerp(top, bottom, wy)));
                tensor.data[channel * plane + dst_index] = @as(f32, @floatFromInt(sampled)) / 255.0;
            }
        }
    }

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

fn computeLetterboxInfo(
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
) LetterboxInfo {
    const scale = @min(
        @as(f32, @floatFromInt(dst_width)) / @as(f32, @floatFromInt(src_width)),
        @as(f32, @floatFromInt(dst_height)) / @as(f32, @floatFromInt(src_height)),
    );
    const resized_width = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src_width)) * scale))));
    const resized_height = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src_height)) * scale))));
    const pad_left = (dst_width - resized_width) / 2;
    const pad_top = (dst_height - resized_height) / 2;

    return .{
        .src_width = src_width,
        .src_height = src_height,
        .dst_width = dst_width,
        .dst_height = dst_height,
        .resized_width = resized_width,
        .resized_height = resized_height,
        .pad_left = pad_left,
        .pad_top = pad_top,
        .scale_x = @as(f32, @floatFromInt(resized_width)) / @as(f32, @floatFromInt(src_width)),
        .scale_y = @as(f32, @floatFromInt(resized_height)) / @as(f32, @floatFromInt(src_height)),
    };
}

fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn clampIndex(value: isize, upper: usize) usize {
    if (value < 0) return 0;
    const upper_index: isize = @intCast(upper - 1);
    if (value > upper_index) return upper - 1;
    return @intCast(value);
}
