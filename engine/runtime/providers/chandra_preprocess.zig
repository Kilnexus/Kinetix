const std = @import("std");
const imaging = @import("Pixio");

pub const SizeConfig = struct {
    longest_edge: usize,
    shortest_edge: usize,
};

pub const ImageProcessorConfig = struct {
    data_format: []const u8 = "channels_first",
    do_convert_rgb: bool = true,
    do_normalize: bool = true,
    do_rescale: bool = true,
    do_resize: bool = true,
    image_mean: []const f64 = &.{ 0.5, 0.5, 0.5 },
    image_processor_type: []const u8 = "Qwen2VLImageProcessorFast",
    image_std: []const f64 = &.{ 0.5, 0.5, 0.5 },
    merge_size: usize,
    patch_size: usize,
    resample: usize = 3,
    rescale_factor: f64 = 1.0 / 255.0,
    size: SizeConfig,
    temporal_patch_size: usize,

    pub fn patchMergeFactor(self: ImageProcessorConfig) usize {
        return self.patch_size * self.merge_size;
    }
};

const ProcessorConfig = struct {
    image_processor: ImageProcessorConfig,
    processor_class: []const u8 = "",
};

pub const ParsedImageProcessorConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: ImageProcessorConfig,

    pub fn deinit(self: *ParsedImageProcessorConfig) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const PatchGrid = struct {
    source_frame_count: usize,
    frame_count: usize,
    temporal_patch_count: usize,
    input_width: usize,
    input_height: usize,
    resized_width: usize,
    resized_height: usize,
    patch_width: usize,
    patch_height: usize,
    patch_token_count: usize,
    merged_width: usize,
    merged_height: usize,
    token_count: usize,

    pub fn patchSize(self: PatchGrid) usize {
        if (self.patch_width == 0) return 0;
        return self.resized_width / self.patch_width;
    }
};

pub const PreparedImageInput = struct {
    tensor: imaging.TensorF32NCHW,
    grid: PatchGrid,

    pub fn deinit(self: *PreparedImageInput) void {
        self.tensor.deinit();
        self.* = undefined;
    }
};

pub fn loadImageProcessorConfig(backing_allocator: std.mem.Allocator, model_path: []const u8) !ParsedImageProcessorConfig {
    const processor_path = try std.fs.path.join(backing_allocator, &.{ model_path, "processor_config.json" });
    defer backing_allocator.free(processor_path);
    if (pathExists(processor_path)) {
        return try loadFromProcessorConfig(backing_allocator, processor_path);
    }

    const preprocessor_path = try std.fs.path.join(backing_allocator, &.{ model_path, "preprocessor_config.json" });
    defer backing_allocator.free(preprocessor_path);
    if (pathExists(preprocessor_path)) {
        return try loadFromPreprocessorConfig(backing_allocator, preprocessor_path);
    }

    return error.MissingChandraPreprocessorConfig;
}

pub fn loadFromPreprocessorConfig(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedImageProcessorConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const config = try std.json.parseFromSliceLeaky(ImageProcessorConfig, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{ .arena = arena, .value = config };
}

pub fn loadFromProcessorConfig(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedImageProcessorConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const wrapper = try std.json.parseFromSliceLeaky(ProcessorConfig, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{ .arena = arena, .value = wrapper.image_processor };
}

pub fn planPatchGrid(config: ImageProcessorConfig, width: usize, height: usize, source_frame_count: usize) !PatchGrid {
    if (width == 0 or height == 0) return error.InvalidImageDimensions;
    if (config.patch_size == 0 or config.merge_size == 0) return error.InvalidImageProcessorConfig;
    if (source_frame_count == 0) return error.InvalidBatchSize;

    const factor = config.patchMergeFactor();
    var resized_width = width;
    var resized_height = height;

    if (config.do_resize) {
        const scale = areaScale(width, height, config.size.shortest_edge, config.size.longest_edge);
        resized_width = roundToMultiple(@max(factor, floatToUsize(@as(f64, @floatFromInt(width)) * scale)), factor);
        resized_height = roundToMultiple(@max(factor, floatToUsize(@as(f64, @floatFromInt(height)) * scale)), factor);
    }

    const patch_width = resized_width / config.patch_size;
    const patch_height = resized_height / config.patch_size;
    const frame_count = roundUpToMultiple(@max(source_frame_count, config.temporal_patch_size), config.temporal_patch_size);
    const temporal_patch_count = frame_count / config.temporal_patch_size;
    const merged_width = patch_width / config.merge_size;
    const merged_height = patch_height / config.merge_size;

    return .{
        .source_frame_count = source_frame_count,
        .frame_count = frame_count,
        .temporal_patch_count = temporal_patch_count,
        .input_width = width,
        .input_height = height,
        .resized_width = resized_width,
        .resized_height = resized_height,
        .patch_width = patch_width,
        .patch_height = patch_height,
        .patch_token_count = temporal_patch_count * patch_width * patch_height,
        .merged_width = merged_width,
        .merged_height = merged_height,
        .token_count = temporal_patch_count * merged_width * merged_height,
    };
}

pub fn loadImageInput(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: ImageProcessorConfig,
) !PreparedImageInput {
    var source = try imaging.decodeFileRgb8(allocator, path);
    defer source.deinit();

    const frames = [_]*const imaging.ImageU8{&source};
    return try prepareImageFramesInput(allocator, &frames, config);
}

pub fn prepareImageInput(
    allocator: std.mem.Allocator,
    source: *const imaging.ImageU8,
    config: ImageProcessorConfig,
) !PreparedImageInput {
    const frames = [_]*const imaging.ImageU8{source};
    return try prepareImageFramesInput(allocator, &frames, config);
}

pub fn prepareImageFramesInput(
    allocator: std.mem.Allocator,
    sources: []const *const imaging.ImageU8,
    config: ImageProcessorConfig,
) !PreparedImageInput {
    if (sources.len == 0) return error.InvalidBatchSize;

    const first = sources[0];
    if (first.channels != 3) return error.InvalidChannelCount;
    for (sources[1..]) |source| {
        if (source.channels != first.channels) return error.InvalidChannelCount;
        if (source.width != first.width or source.height != first.height) return error.ShapeMismatch;
    }

    const grid = try planPatchGrid(config, first.width, first.height, sources.len);
    var tensor = imaging.TensorF32NCHW{
        .allocator = allocator,
        .batch = grid.frame_count,
        .channels = first.channels,
        .height = grid.resized_height,
        .width = grid.resized_width,
        .stride_w = 1,
        .stride_h = grid.resized_width,
        .stride_c = grid.resized_width * grid.resized_height,
        .stride_n = first.channels * grid.resized_width * grid.resized_height,
        .data = try allocator.alloc(f32, grid.frame_count * first.channels * grid.resized_width * grid.resized_height),
    };
    errdefer tensor.deinit();

    var mean_storage: [4]f32 = undefined;
    var std_storage: [4]f32 = undefined;
    const mean = if (config.do_normalize)
        try statsF32(config.image_mean, &mean_storage, first.channels)
    else
        &.{};
    const stddev = if (config.do_normalize)
        try statsF32(config.image_std, &std_storage, first.channels)
    else
        &.{};

    for (0..grid.frame_count) |frame_index| {
        const source = sources[@min(frame_index, sources.len - 1)];
        var resized: ?imaging.ImageU8 = null;
        defer if (resized) |*image| image.deinit();

        const tensor_source = if (grid.resized_width != source.width or grid.resized_height != source.height) blk: {
            resized = try imaging.resizeImage(
                allocator,
                source,
                grid.resized_width,
                grid.resized_height,
                resizeKernel(config.resample),
            );
            break :blk &resized.?;
        } else source;

        var chw_tensor = try imaging.imageToTensorChwF32(allocator, tensor_source, .{
            .scale = if (config.do_rescale) @floatCast(config.rescale_factor) else 1.0,
            .mean = mean,
            .std = stddev,
        });
        defer chw_tensor.deinit();

        const dst = tensor.data[frame_index * tensor.stride_n ..][0..tensor.stride_n];
        @memcpy(dst, chw_tensor.data);
    }

    return .{
        .tensor = tensor,
        .grid = grid,
    };
}

fn areaScale(width: usize, height: usize, min_pixels: usize, max_pixels: usize) f64 {
    const area = @as(f64, @floatFromInt(width)) * @as(f64, @floatFromInt(height));
    const min_area = @as(f64, @floatFromInt(min_pixels));
    const max_area = @as(f64, @floatFromInt(max_pixels));

    if (area > max_area) return @sqrt(max_area / area);
    if (area < min_area) return @sqrt(min_area / area);
    return 1.0;
}

fn floatToUsize(value: f64) usize {
    if (value <= 1.0) return 1;
    return @as(usize, @intFromFloat(@round(value)));
}

fn roundToMultiple(value: usize, multiple: usize) usize {
    const rounded = ((value + multiple / 2) / multiple) * multiple;
    return @max(multiple, rounded);
}

fn roundUpToMultiple(value: usize, multiple: usize) usize {
    return ((value + multiple - 1) / multiple) * multiple;
}

fn resizeKernel(resample: usize) imaging.ResizeKernel {
    return switch (resample) {
        0 => .nearest,
        1 => .lanczos3,
        2 => .bilinear,
        3 => .bicubic,
        else => .bilinear,
    };
}

fn statsF32(values: []const f64, storage: *[4]f32, channels: usize) ![]const f32 {
    if (values.len != 1 and values.len != channels) return error.InvalidNormalizationSpec;

    const len = if (values.len == 1) 1 else channels;
    for (0..len) |index| {
        storage[index] = @floatCast(values[index]);
    }
    return storage[0..len];
}

fn pathExists(path: []const u8) bool {
    const file = std.fs.openFileAbsolute(path, .{}) catch return false;
    file.close();
    return true;
}

test "chandra preprocessor config parser accepts official qwen image processor shape" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "data_format": "channels_first",
        \\  "do_convert_rgb": true,
        \\  "do_normalize": true,
        \\  "do_rescale": true,
        \\  "do_resize": true,
        \\  "image_mean": [0.5, 0.5, 0.5],
        \\  "image_processor_type": "Qwen2VLImageProcessorFast",
        \\  "image_std": [0.5, 0.5, 0.5],
        \\  "merge_size": 2,
        \\  "patch_size": 16,
        \\  "resample": 3,
        \\  "rescale_factor": 0.00392156862745098,
        \\  "size": {
        \\    "longest_edge": 16777216,
        \\    "shortest_edge": 65536
        \\  },
        \\  "temporal_patch_size": 2
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "preprocessor_config.json");
    defer std.testing.allocator.free(path);

    var parsed = try loadFromPreprocessorConfig(std.testing.allocator, path);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 16), parsed.value.patch_size);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.merge_size);
    try std.testing.expectEqual(@as(usize, 65536), parsed.value.size.shortest_edge);

    const grid = try planPatchGrid(parsed.value, 1024, 768, 1);
    try std.testing.expectEqual(@as(usize, 1024), grid.resized_width);
    try std.testing.expectEqual(@as(usize, 768), grid.resized_height);
    try std.testing.expectEqual(@as(usize, 1), grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), grid.frame_count);
    try std.testing.expectEqual(@as(usize, 1), grid.temporal_patch_count);
    try std.testing.expectEqual(@as(usize, 64), grid.patch_width);
    try std.testing.expectEqual(@as(usize, 64 * 48), grid.patch_token_count);
    try std.testing.expectEqual(@as(usize, 24 * 32), grid.token_count);
}

test "chandra processor config parser prefers nested image_processor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "processor_config.json",
        \\{
        \\  "processor_class": "Qwen3VLProcessor",
        \\  "image_processor": {
        \\    "merge_size": 2,
        \\    "patch_size": 16,
        \\    "temporal_patch_size": 2,
        \\    "size": {
        \\      "longest_edge": 16777216,
        \\      "shortest_edge": 65536
        \\    }
        \\  }
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "processor_config.json");
    defer std.testing.allocator.free(path);

    var parsed = try loadFromProcessorConfig(std.testing.allocator, path);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 16), parsed.value.patch_size);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.temporal_patch_size);
}

test "chandra image preprocessing produces normalized chw tensor and patch grid" {
    var image = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image.deinit();
    image.fill(255);

    var prepared = try prepareImageInput(std.testing.allocator, &image, .{
        .merge_size = 2,
        .patch_size = 16,
        .temporal_patch_size = 2,
        .resample = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1024,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 32), prepared.grid.resized_width);
    try std.testing.expectEqual(@as(usize, 32), prepared.grid.resized_height);
    try std.testing.expectEqual(@as(usize, 4), prepared.grid.patch_token_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.token_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.temporal_patch_count);
    try std.testing.expectEqual(@as(usize, 3), prepared.tensor.channels);
    try std.testing.expectEqual(@as(usize, 2), prepared.tensor.batch);
    try std.testing.expectEqual(@as(usize, 32), prepared.tensor.width);
    try std.testing.expectEqual(@as(usize, 32), prepared.tensor.height);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), prepared.tensor.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), prepared.tensor.data[prepared.tensor.stride_n], 0.0001);
}

test "chandra frame preprocessing pads temporal batch and preserves distinct frames" {
    var image_a = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image_a.deinit();
    image_a.fill(0);

    var image_b = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image_b.deinit();
    image_b.fill(255);

    const frames = [_]*const imaging.ImageU8{ &image_a, &image_b };
    var prepared = try prepareImageFramesInput(std.testing.allocator, &frames, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 2), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.temporal_patch_count);
    try std.testing.expectEqual(@as(usize, 4), prepared.grid.patch_token_count);
    try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[0]);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
