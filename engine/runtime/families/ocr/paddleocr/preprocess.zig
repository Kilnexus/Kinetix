const std = @import("std");
const imaging = @import("Pixio");
const Tensor = @import("shared_graph").runtime.tensor.Tensor;

pub const Mode = enum {
    stretch,
    recognition,
};

pub const ShapeStage = enum {
    det,
    rec,
    cls,
    unknown,
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

pub const ShapeResolveOptions = struct {
    stage: ShapeStage = .unknown,
    image_width: usize = 0,
    image_height: usize = 0,
    det_max_side: usize = 960,
    rec_height: usize = 48,
    rec_max_width: usize = 320,
    cls_height: usize = 48,
    cls_width: usize = 192,
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

pub fn shapeFromModelInputForImage(
    allocator: std.mem.Allocator,
    dims: []const @import("shared_graph").onnx.metadata.Dimension,
    options: ShapeResolveOptions,
) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(shape);
    if (shape.len != 4) return error.UnsupportedTensorRank;
    if (options.image_width == 0 or options.image_height == 0) return error.InvalidImageDimensions;

    shape[0] = try resolveDim(dims[0], 1);
    shape[1] = try resolveDim(dims[1], 3);
    if (shape[0] != 1 or shape[1] != 3) return error.UnsupportedTensorShape;

    const fallback = fallbackSpatialShape(options);
    shape[2] = try resolveDim(dims[2], fallback.height);
    shape[3] = try resolveDim(dims[3], fallback.width);
    if (shape[2] == 0 or shape[3] == 0) return error.InvalidImageDimensions;
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

const SpatialShape = struct {
    width: usize,
    height: usize,
};

fn resolveDim(dim: @import("shared_graph").onnx.metadata.Dimension, fallback: usize) !usize {
    return switch (dim) {
        .value => |value| if (value > 0) @intCast(value) else fallback,
        else => fallback,
    };
}

fn fallbackSpatialShape(options: ShapeResolveOptions) SpatialShape {
    return switch (options.stage) {
        .det => detectionShape(options.image_width, options.image_height, options.det_max_side),
        .rec => recognitionShape(options.image_width, options.image_height, options.rec_height, options.rec_max_width),
        .cls => .{ .width = options.cls_width, .height = options.cls_height },
        .unknown => .{ .width = options.image_width, .height = options.image_height },
    };
}

fn detectionShape(width: usize, height: usize, max_side: usize) SpatialShape {
    var out_w = width;
    var out_h = height;
    const long_side = @max(width, height);
    if (max_side != 0 and long_side > max_side) {
        out_w = @max(@as(usize, 1), (width * max_side) / long_side);
        out_h = @max(@as(usize, 1), (height * max_side) / long_side);
    }
    return .{
        .width = alignUp(out_w, 32),
        .height = alignUp(out_h, 32),
    };
}

fn recognitionShape(width: usize, height: usize, target_height: usize, max_width: usize) SpatialShape {
    const scaled_w = @max(@as(usize, 1), (width * target_height + height - 1) / height);
    return .{
        .width = alignUp(@min(max_width, scaled_w), 32),
        .height = target_height,
    };
}

fn alignUp(value: usize, alignment: usize) usize {
    if (alignment == 0) return value;
    return ((value + alignment - 1) / alignment) * alignment;
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

test "paddleocr dynamic input shape resolves by stage" {
    const Dimension = @import("shared_graph").onnx.metadata.Dimension;
    const shape = try shapeFromModelInputForImage(
        std.testing.allocator,
        &.{ Dimension{ .value = -1 }, Dimension{ .value = 3 }, Dimension{ .param = @constCast("h") }, Dimension{ .param = @constCast("w") } },
        .{ .stage = .det, .image_width = 641, .image_height = 320 },
    );
    defer std.testing.allocator.free(shape);

    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 320, 672 }, shape);

    const rec_shape = try shapeFromModelInputForImage(
        std.testing.allocator,
        &.{ Dimension{ .value = 1 }, Dimension{ .value = 3 }, Dimension{ .param = @constCast("h") }, Dimension{ .param = @constCast("w") } },
        .{ .stage = .rec, .image_width = 120, .image_height = 24 },
    );
    defer std.testing.allocator.free(rec_shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 48, 256 }, rec_shape);
}
