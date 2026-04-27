const std = @import("std");
const imaging = @import("Pixio");
const Tensor = @import("shared_graph").runtime.tensor.Tensor;

pub const Options = struct {
    normalize: imaging.NormalizeOptions = .{
        .scale = 1.0 / 255.0,
        .mean = &.{ 0.5, 0.5, 0.5 },
        .std = &.{ 0.5, 0.5, 0.5 },
    },
    kernel: imaging.ResizeKernel = .bilinear,
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
