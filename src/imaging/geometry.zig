pub const BoxF32 = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
};

pub fn remapLetterboxedBoxToSource(
    box: *BoxF32,
    pad_left: usize,
    pad_top: usize,
    scale_x: f32,
    scale_y: f32,
    src_width: usize,
    src_height: usize,
) void {
    const left = @as(f32, @floatFromInt(pad_left));
    const top = @as(f32, @floatFromInt(pad_top));
    const width = @as(f32, @floatFromInt(src_width));
    const height = @as(f32, @floatFromInt(src_height));

    box.x1 = clipToRange((box.x1 - left) / scale_x, 0.0, width);
    box.y1 = clipToRange((box.y1 - top) / scale_y, 0.0, height);
    box.x2 = clipToRange((box.x2 - left) / scale_x, 0.0, width);
    box.y2 = clipToRange((box.y2 - top) / scale_y, 0.0, height);
}

fn clipToRange(value: f32, min_value: f32, max_value: f32) f32 {
    return @max(min_value, @min(max_value, value));
}
