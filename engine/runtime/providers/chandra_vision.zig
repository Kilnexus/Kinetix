const std = @import("std");
const imaging = @import("Pixio");
const preprocess = @import("chandra_preprocess.zig");

pub const PatchEmbeddingWeights = struct {
    data: []const f32,
    bias: ?[]const f32 = null,
    out_channels: usize,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,

    pub fn expectedLen(self: PatchEmbeddingWeights) usize {
        return self.out_channels * self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
    }
};

pub const OwnedPatchEmbeddingWeights = struct {
    allocator: std.mem.Allocator,
    weights: PatchEmbeddingWeights,

    pub fn deinit(self: *OwnedPatchEmbeddingWeights) void {
        self.allocator.free(self.weights.data);
        if (self.weights.bias) |bias| self.allocator.free(bias);
        self.* = undefined;
    }
};

pub const PatchEmbeddings = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    token_count: usize,
    embedding_dim: usize,
    patch_width: usize,
    patch_height: usize,

    pub fn deinit(self: *PatchEmbeddings) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn at(self: PatchEmbeddings, token_index: usize, channel: usize) f32 {
        return self.data[token_index * self.embedding_dim + channel];
    }
};

pub const MergedPatchGroups = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    token_count: usize,
    merged_dim: usize,
    merged_width: usize,
    merged_height: usize,

    pub fn deinit(self: *MergedPatchGroups) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn at(self: MergedPatchGroups, token_index: usize, offset: usize) f32 {
        return self.data[token_index * self.merged_dim + offset];
    }
};

pub fn patchEmbedImage(
    allocator: std.mem.Allocator,
    input: *const preprocess.PreparedImageInput,
    weights: PatchEmbeddingWeights,
) !PatchEmbeddings {
    try validatePatchEmbeddingInputs(input, weights);

    var output = PatchEmbeddings{
        .allocator = allocator,
        .data = try allocator.alloc(f32, input.grid.patch_token_count * weights.out_channels),
        .token_count = input.grid.patch_token_count,
        .embedding_dim = weights.out_channels,
        .patch_width = input.grid.patch_width,
        .patch_height = input.grid.patch_height,
    };
    errdefer output.deinit();

    for (0..input.grid.patch_height) |patch_y| {
        for (0..input.grid.patch_width) |patch_x| {
            const token_index = patch_y * input.grid.patch_width + patch_x;
            for (0..weights.out_channels) |out_channel| {
                var sum = if (weights.bias) |bias| bias[out_channel] else 0.0;
                for (0..weights.in_channels) |in_channel| {
                    for (0..weights.temporal_patch_size) |temporal_index| {
                        for (0..weights.patch_size) |kernel_y| {
                            const image_y = patch_y * weights.patch_size + kernel_y;
                            for (0..weights.patch_size) |kernel_x| {
                                const image_x = patch_x * weights.patch_size + kernel_x;
                                const pixel = input.tensor.data[tensorIndex(input.tensor, in_channel, image_y, image_x)];
                                const weight = weights.data[weightIndex(weights, out_channel, in_channel, temporal_index, kernel_y, kernel_x)];
                                sum += pixel * weight;
                            }
                        }
                    }
                }
                output.data[token_index * weights.out_channels + out_channel] = sum;
            }
        }
    }

    return output;
}

pub fn mergeSpatialPatches(
    allocator: std.mem.Allocator,
    embeddings: PatchEmbeddings,
    merge_size: usize,
) !MergedPatchGroups {
    if (merge_size == 0) return error.InvalidMergeSize;
    if (embeddings.patch_width % merge_size != 0 or embeddings.patch_height % merge_size != 0) return error.ShapeMismatch;

    const merged_width = embeddings.patch_width / merge_size;
    const merged_height = embeddings.patch_height / merge_size;
    const patches_per_group = merge_size * merge_size;
    const merged_dim = embeddings.embedding_dim * patches_per_group;

    var output = MergedPatchGroups{
        .allocator = allocator,
        .data = try allocator.alloc(f32, merged_width * merged_height * merged_dim),
        .token_count = merged_width * merged_height,
        .merged_dim = merged_dim,
        .merged_width = merged_width,
        .merged_height = merged_height,
    };
    errdefer output.deinit();

    for (0..merged_height) |group_y| {
        for (0..merged_width) |group_x| {
            const merged_index = group_y * merged_width + group_x;
            const merged_base = merged_index * merged_dim;

            for (0..merge_size) |inner_y| {
                for (0..merge_size) |inner_x| {
                    const patch_y = group_y * merge_size + inner_y;
                    const patch_x = group_x * merge_size + inner_x;
                    const patch_index = patch_y * embeddings.patch_width + patch_x;
                    const patch_slot = inner_y * merge_size + inner_x;
                    const dst_base = merged_base + patch_slot * embeddings.embedding_dim;
                    const src_base = patch_index * embeddings.embedding_dim;
                    @memcpy(
                        output.data[dst_base .. dst_base + embeddings.embedding_dim],
                        embeddings.data[src_base .. src_base + embeddings.embedding_dim],
                    );
                }
            }
        }
    }

    return output;
}

fn validatePatchEmbeddingInputs(input: *const preprocess.PreparedImageInput, weights: PatchEmbeddingWeights) !void {
    if (weights.out_channels == 0 or weights.in_channels == 0) return error.InvalidPatchEmbeddingShape;
    if (weights.patch_size == 0 or weights.temporal_patch_size == 0) return error.InvalidPatchEmbeddingShape;
    if (weights.data.len != weights.expectedLen()) return error.ShapeMismatch;
    if (weights.bias) |bias| {
        if (bias.len != weights.out_channels) return error.ShapeMismatch;
    }
    if (input.tensor.channels != weights.in_channels) return error.InvalidChannelCount;
    if (input.grid.patchSize() != weights.patch_size) return error.ShapeMismatch;
    if (input.grid.resized_width != input.tensor.width or input.grid.resized_height != input.tensor.height) return error.ShapeMismatch;
}

fn tensorIndex(tensor: imaging.TensorF32CHW, channel: usize, y: usize, x: usize) usize {
    return channel * tensor.width * tensor.height + y * tensor.width + x;
}

fn weightIndex(
    weights: PatchEmbeddingWeights,
    out_channel: usize,
    in_channel: usize,
    temporal_index: usize,
    y: usize,
    x: usize,
) usize {
    return (((out_channel * weights.in_channels + in_channel) * weights.temporal_patch_size + temporal_index) * weights.patch_size + y) * weights.patch_size + x;
}

test "chandra patch embedding projects image patches into token embeddings" {
    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();

    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }

    var prepared = try preprocess.prepareImageInput(std.testing.allocator, &image, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 2,
        .temporal_patch_size = 1,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const weights = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    const bias = [_]f32{ 0, 1 };

    var embeddings = try patchEmbedImage(std.testing.allocator, &prepared, .{
        .data = &weights,
        .bias = &bias,
        .out_channels = 2,
        .in_channels = 3,
        .temporal_patch_size = 1,
        .patch_size = 2,
    });
    defer embeddings.deinit();

    try std.testing.expectEqual(@as(usize, 2), embeddings.token_count);
    try std.testing.expectEqual(@as(usize, 2), embeddings.embedding_dim);
    try std.testing.expectEqual(@as(f32, 14), embeddings.at(0, 0));
    try std.testing.expectEqual(@as(f32, 29), embeddings.at(0, 1));
    try std.testing.expectEqual(@as(f32, 22), embeddings.at(1, 0));
    try std.testing.expectEqual(@as(f32, 45), embeddings.at(1, 1));
}

test "chandra patch merger groups neighboring patch embeddings" {
    var embeddings = PatchEmbeddings{
        .allocator = std.testing.allocator,
        .data = try std.testing.allocator.dupe(f32, &.{
            1, 10,
            2, 20,
            3, 30,
            4, 40,
        }),
        .token_count = 4,
        .embedding_dim = 2,
        .patch_width = 2,
        .patch_height = 2,
    };
    defer embeddings.deinit();

    var merged = try mergeSpatialPatches(std.testing.allocator, embeddings, 2);
    defer merged.deinit();

    try std.testing.expectEqual(@as(usize, 1), merged.token_count);
    try std.testing.expectEqual(@as(usize, 8), merged.merged_dim);
    try std.testing.expectEqual(@as(f32, 1), merged.at(0, 0));
    try std.testing.expectEqual(@as(f32, 10), merged.at(0, 1));
    try std.testing.expectEqual(@as(f32, 2), merged.at(0, 2));
    try std.testing.expectEqual(@as(f32, 20), merged.at(0, 3));
    try std.testing.expectEqual(@as(f32, 3), merged.at(0, 4));
    try std.testing.expectEqual(@as(f32, 30), merged.at(0, 5));
    try std.testing.expectEqual(@as(f32, 4), merged.at(0, 6));
    try std.testing.expectEqual(@as(f32, 40), merged.at(0, 7));
}
