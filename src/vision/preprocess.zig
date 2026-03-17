const std = @import("std");
const runtime = @import("runtime");
const image = @import("image.zig");

pub const LetterboxInfo = struct {
    src_width: usize,
    src_height: usize,
    input_size: usize,
    resized_width: usize,
    resized_height: usize,
    pad_left: usize,
    pad_top: usize,
    scale_x: f32,
    scale_y: f32,
};

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

    var tensor = try runtime.Tensor.init(allocator, 1, 3, input_size, input_size);
    errdefer tensor.deinit();
    tensor.fill(114.0 / 255.0);

    const scale = @min(
        @as(f32, @floatFromInt(input_size)) / @as(f32, @floatFromInt(src.width)),
        @as(f32, @floatFromInt(input_size)) / @as(f32, @floatFromInt(src.height)),
    );
    const resized_width = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src.width)) * scale))));
    const resized_height = @max(@as(usize, 1), @as(usize, @intFromFloat(@round(@as(f32, @floatFromInt(src.height)) * scale))));
    const pad_left = (input_size - resized_width) / 2;
    const pad_top = (input_size - resized_height) / 2;
    const scale_x = @as(f32, @floatFromInt(resized_width)) / @as(f32, @floatFromInt(src.width));
    const scale_y = @as(f32, @floatFromInt(resized_height)) / @as(f32, @floatFromInt(src.height));

    for (0..resized_height) |dy| {
        const src_y = ((@as(f32, @floatFromInt(dy)) + 0.5) * @as(f32, @floatFromInt(src.height)) / @as(f32, @floatFromInt(resized_height))) - 0.5;
        const y0 = clampIndex(@as(isize, @intFromFloat(@floor(src_y))), src.height);
        const y1 = if (y0 + 1 < src.height) y0 + 1 else src.height - 1;
        const wy = src_y - @as(f32, @floatFromInt(y0));

        for (0..resized_width) |dx| {
            const src_x = ((@as(f32, @floatFromInt(dx)) + 0.5) * @as(f32, @floatFromInt(src.width)) / @as(f32, @floatFromInt(resized_width))) - 0.5;
            const x0 = clampIndex(@as(isize, @intFromFloat(@floor(src_x))), src.width);
            const x1 = if (x0 + 1 < src.width) x0 + 1 else src.width - 1;
            const wx = src_x - @as(f32, @floatFromInt(x0));

            const out_y = pad_top + dy;
            const out_x = pad_left + dx;

            for (0..3) |channel| {
                const p00 = pixelAt(&src, x0, y0, channel);
                const p10 = pixelAt(&src, x1, y0, channel);
                const p01 = pixelAt(&src, x0, y1, channel);
                const p11 = pixelAt(&src, x1, y1, channel);

                const top = lerp(p00, p10, wx);
                const bottom = lerp(p01, p11, wx);
                const value = lerp(top, bottom, wy) / 255.0;
                tensor.set(0, channel, out_y, out_x, value);
            }
        }
    }

    return .{
        .tensor = tensor,
        .info = .{
            .src_width = src.width,
            .src_height = src.height,
            .input_size = input_size,
            .resized_width = resized_width,
            .resized_height = resized_height,
            .pad_left = pad_left,
            .pad_top = pad_top,
            .scale_x = scale_x,
            .scale_y = scale_y,
        },
    };
}

pub fn remapDetectionsToSource(detections: []runtime.Detection, info: LetterboxInfo) void {
    const pad_left = @as(f32, @floatFromInt(info.pad_left));
    const pad_top = @as(f32, @floatFromInt(info.pad_top));
    const src_width = @as(f32, @floatFromInt(info.src_width));
    const src_height = @as(f32, @floatFromInt(info.src_height));

    for (detections) |*det| {
        det.x1 = clipToRange((det.x1 - pad_left) / info.scale_x, 0.0, src_width);
        det.y1 = clipToRange((det.y1 - pad_top) / info.scale_y, 0.0, src_height);
        det.x2 = clipToRange((det.x2 - pad_left) / info.scale_x, 0.0, src_width);
        det.y2 = clipToRange((det.y2 - pad_top) / info.scale_y, 0.0, src_height);
    }
}

fn pixelAt(src: *const image.ImageU8, x: usize, y: usize, channel: usize) f32 {
    const index = (y * src.width + x) * src.channels + channel;
    return @floatFromInt(src.data[index]);
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

fn clipToRange(value: f32, min_value: f32, max_value: f32) f32 {
    return @max(min_value, @min(max_value, value));
}
