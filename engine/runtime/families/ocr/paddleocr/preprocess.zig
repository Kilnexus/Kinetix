const std = @import("std");
const imaging = @import("Pixio");
const Tensor = @import("shared_graph").runtime.tensor.Tensor;

pub const Mode = enum {
    stretch,
    recognition,
};

pub const Options = struct {
    normalize: imaging.NormalizeOptions = .{
        .scale = 1.0 / 255.0,
        .mean = &.{ 0.5, 0.5, 0.5 },
        .std = &.{ 0.5, 0.5, 0.5 },
    },
    kernel: imaging.ResizeKernel = .bilinear,
    mode: Mode = .stretch,
};

pub fn tensorFromImagePath(
    allocator: std.mem.Allocator,
    path: []const u8,
    nchw_shape: []const usize,
    options: Options,
) !Tensor {
    var image = try imaging.decodeFileRgb8(allocator, path);
    defer image.deinit();
    return try tensorFromImage(allocator, &image, nchw_shape, options);
}

pub fn tensorFromImage(
    allocator: std.mem.Allocator,
    image: *const imaging.ImageU8,
    nchw_shape: []const usize,
    options: Options,
) !Tensor {
    if (nchw_shape.len != 4) return error.UnsupportedTensorRank;
    if (nchw_shape[0] != 1) return error.UnsupportedBatchSize;
    if (nchw_shape[1] != 3) return error.InvalidChannelCount;
    if (image.channels != 3) return error.InvalidChannelCount;
    try validateStats(image.channels, options.normalize.mean);
    try validateStats(image.channels, options.normalize.std);

    if (options.mode == .recognition) {
        return try recognitionTensorFromImage(allocator, image, nchw_shape, options);
    }

    const target_h = nchw_shape[2];
    const target_w = nchw_shape[3];
    if (target_w == 0 or target_h == 0) return error.InvalidImageDimensions;

    var resized: ?imaging.ImageU8 = null;
    defer if (resized) |*item| item.deinit();

    const prepared = if (image.width == target_w and image.height == target_h)
        image
    else blk: {
        resized = try imaging.resizeImage(allocator, image, target_w, target_h, options.kernel);
        break :blk &resized.?;
    };

    var chw = try imaging.imageToTensorChwF32(allocator, prepared, options.normalize);
    defer chw.deinit();

    return try Tensor.fromF32(allocator, nchw_shape, chw.data);
}

pub fn recognitionTensorFromImage(
    allocator: std.mem.Allocator,
    image: *const imaging.ImageU8,
    nchw_shape: []const usize,
    options: Options,
) !Tensor {
    if (nchw_shape.len != 4) return error.UnsupportedTensorRank;
    if (nchw_shape[0] != 1) return error.UnsupportedBatchSize;
    if (nchw_shape[1] != 3) return error.InvalidChannelCount;
    if (image.channels != 3) return error.InvalidChannelCount;
    try validateStats(image.channels, options.normalize.mean);
    try validateStats(image.channels, options.normalize.std);

    const target_h = nchw_shape[2];
    const target_w = nchw_shape[3];
    if (target_w == 0 or target_h == 0) return error.InvalidImageDimensions;

    const resized_w = recognitionResizeWidth(image.width, image.height, target_w, target_h);
    var resized = try imaging.resizeImage(allocator, image, resized_w, target_h, options.kernel);
    defer resized.deinit();

    const owned_shape = try allocator.dupe(usize, nchw_shape);
    errdefer allocator.free(owned_shape);
    const values = try allocator.alloc(f32, nchw_shape[0] * nchw_shape[1] * target_h * target_w);
    errdefer allocator.free(values);
    @memset(values, 0.0);

    for (0..target_h) |y| {
        for (0..resized_w) |x| {
            for (0..3) |channel| {
                values[channel * target_h * target_w + y * target_w + x] = normalizePixel(
                    resized.get(x, y, channel),
                    channel,
                    options.normalize,
                );
            }
        }
    }

    return .{
        .allocator = allocator,
        .shape = owned_shape,
        .buffer = .{ .f32 = values },
    };
}

fn recognitionResizeWidth(src_width: usize, src_height: usize, target_width: usize, target_height: usize) usize {
    const scaled = (target_height * src_width + src_height - 1) / src_height;
    return @max(@as(usize, 1), @min(target_width, scaled));
}

fn normalizePixel(value: u8, channel: usize, options: imaging.NormalizeOptions) f32 {
    const std_value = statValueOrOne(options.std, channel);
    return (@as(f32, @floatFromInt(value)) * options.scale - statValue(options.mean, channel)) / std_value;
}

fn statValue(values: []const f32, channel: usize) f32 {
    if (values.len == 0) return 0.0;
    if (values.len == 1) return values[0];
    return values[channel];
}

fn statValueOrOne(values: []const f32, channel: usize) f32 {
    if (values.len == 0) return 1.0;
    if (values.len == 1) return values[0];
    return values[channel];
}

fn validateStats(channels: usize, values: []const f32) !void {
    if (values.len == 0 or values.len == 1 or values.len == channels) return;
    return error.InvalidNormalizationSpec;
}

pub fn shapeFromModelInput(allocator: std.mem.Allocator, dims: []const @import("shared_graph").onnx.metadata.Dimension) ![]usize {
    const shape = try dimsToShape(allocator, dims);
    errdefer allocator.free(shape);
    if (shape.len != 4) return error.UnsupportedTensorRank;
    if (shape[0] != 1 or shape[1] != 3) return error.UnsupportedTensorShape;
    return shape;
}

fn dimsToShape(allocator: std.mem.Allocator, dims: []const @import("shared_graph").onnx.metadata.Dimension) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(shape);
    for (dims, shape) |dim, *slot| {
        slot.* = switch (dim) {
            .value => |value| if (value <= 0) return error.DynamicTensorShape else @intCast(value),
            else => return error.DynamicTensorShape,
        };
    }
    return shape;
}

test "paddleocr preprocess creates normalized nchw tensor" {
    var image = try imaging.ImageU8.init(std.testing.allocator, 2, 1, 3);
    defer image.deinit();
    image.data[0] = 255;
    image.data[1] = 255;
    image.data[2] = 255;
    image.data[3] = 0;
    image.data[4] = 0;
    image.data[5] = 0;

    var tensor = try tensorFromImage(std.testing.allocator, &image, &.{ 1, 3, 1, 2 }, .{});
    defer tensor.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 1, 2 }, tensor.shape);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tensor.buffer.f32[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), tensor.buffer.f32[1], 0.0001);
}

test "paddleocr recognition preprocess keeps aspect ratio and pads tensor with zero" {
    var image = try imaging.ImageU8.init(std.testing.allocator, 2, 1, 3);
    defer image.deinit();
    image.fill(255);

    var tensor = try tensorFromImage(std.testing.allocator, &image, &.{ 1, 3, 2, 8 }, .{ .mode = .recognition });
    defer tensor.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 2, 8 }, tensor.shape);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tensor.buffer.f32[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tensor.buffer.f32[4], 0.0001);
}
