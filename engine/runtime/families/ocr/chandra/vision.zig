const std = @import("std");
const imaging = @import("Pixio");
const preprocess = @import("preprocess.zig");
const attention = @import("../../../text/attention/attention.zig");
const cpu = @import("../../../text/core/cpu.zig");

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
    temporal_patch_count: usize,
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
    temporal_patch_count: usize,
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

pub const LinearWeights = struct {
    data: []const f32,
    bias: ?[]const f32 = null,
    out_features: usize,
    in_features: usize,

    pub fn expectedLen(self: LinearWeights) usize {
        return self.out_features * self.in_features;
    }
};

pub const VisualMergerWeights = struct {
    fc1: LinearWeights,
    fc2: LinearWeights,
};

pub const OwnedVisualMergerWeights = struct {
    allocator: std.mem.Allocator,
    weights: VisualMergerWeights,

    pub fn deinit(self: *OwnedVisualMergerWeights) void {
        self.allocator.free(self.weights.fc1.data);
        if (self.weights.fc1.bias) |bias| self.allocator.free(bias);
        self.allocator.free(self.weights.fc2.data);
        if (self.weights.fc2.bias) |bias| self.allocator.free(bias);
        self.* = undefined;
    }
};

pub const VisualTokens = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    token_count: usize,
    embedding_dim: usize,
    grid_time: usize,
    grid_width: usize,
    grid_height: usize,

    pub fn deinit(self: *VisualTokens) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn at(self: VisualTokens, token_index: usize, channel: usize) f32 {
        return self.data[token_index * self.embedding_dim + channel];
    }
};

pub const LayerNormWeights = struct {
    weight: []const f32,
    bias: []const f32,
    dim: usize,
};

pub const VisualAttentionWeights = struct {
    norm: LayerNormWeights,
    qkv: LinearWeights,
    proj: LinearWeights,
    num_heads: usize,
};

pub const OwnedVisualAttentionWeights = struct {
    allocator: std.mem.Allocator,
    weights: VisualAttentionWeights,

    pub fn deinit(self: *OwnedVisualAttentionWeights) void {
        self.allocator.free(self.weights.norm.weight);
        self.allocator.free(self.weights.norm.bias);
        self.allocator.free(self.weights.qkv.data);
        if (self.weights.qkv.bias) |bias| self.allocator.free(bias);
        self.allocator.free(self.weights.proj.data);
        if (self.weights.proj.bias) |bias| self.allocator.free(bias);
        self.* = undefined;
    }
};

pub const PositionEmbeddingWeights = struct {
    data: []const f32,
    base_width: usize,
    base_height: usize,
    embedding_dim: usize,

    pub fn tokenCount(self: PositionEmbeddingWeights) usize {
        return self.base_width * self.base_height;
    }
};

pub const OwnedPositionEmbeddingWeights = struct {
    allocator: std.mem.Allocator,
    weights: PositionEmbeddingWeights,

    pub fn deinit(self: *OwnedPositionEmbeddingWeights) void {
        self.allocator.free(self.weights.data);
        self.* = undefined;
    }
};

pub const LinearMlpWeights = struct {
    norm: LayerNormWeights,
    fc1: LinearWeights,
    fc2: LinearWeights,
};

pub const OwnedLinearMlpWeights = struct {
    allocator: std.mem.Allocator,
    weights: LinearMlpWeights,

    pub fn deinit(self: *OwnedLinearMlpWeights) void {
        self.allocator.free(self.weights.norm.weight);
        self.allocator.free(self.weights.norm.bias);
        self.allocator.free(self.weights.fc1.data);
        if (self.weights.fc1.bias) |bias| self.allocator.free(bias);
        self.allocator.free(self.weights.fc2.data);
        if (self.weights.fc2.bias) |bias| self.allocator.free(bias);
        self.* = undefined;
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
        .temporal_patch_count = input.grid.temporal_patch_count,
        .patch_width = input.grid.patch_width,
        .patch_height = input.grid.patch_height,
    };
    errdefer output.deinit();

    for (0..input.grid.temporal_patch_count) |temporal_group| {
        for (0..input.grid.patch_height) |patch_y| {
            for (0..input.grid.patch_width) |patch_x| {
                const token_index = (temporal_group * input.grid.patch_height + patch_y) * input.grid.patch_width + patch_x;
                for (0..weights.out_channels) |out_channel| {
                    var sum = if (weights.bias) |bias| bias[out_channel] else 0.0;
                    for (0..weights.in_channels) |in_channel| {
                        for (0..weights.temporal_patch_size) |temporal_index| {
                            for (0..weights.patch_size) |kernel_y| {
                                const image_y = patch_y * weights.patch_size + kernel_y;
                                for (0..weights.patch_size) |kernel_x| {
                                    const image_x = patch_x * weights.patch_size + kernel_x;
                                    const frame_index = temporal_group * weights.temporal_patch_size + temporal_index;
                                    const pixel = input.tensor.data[tensorIndex(input.tensor, frame_index, in_channel, image_y, image_x)];
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
        .data = try allocator.alloc(f32, embeddings.temporal_patch_count * merged_width * merged_height * merged_dim),
        .token_count = embeddings.temporal_patch_count * merged_width * merged_height,
        .merged_dim = merged_dim,
        .temporal_patch_count = embeddings.temporal_patch_count,
        .merged_width = merged_width,
        .merged_height = merged_height,
    };
    errdefer output.deinit();

    for (0..embeddings.temporal_patch_count) |temporal_group| {
        for (0..merged_height) |group_y| {
            for (0..merged_width) |group_x| {
                const merged_index = (temporal_group * merged_height + group_y) * merged_width + group_x;
                const merged_base = merged_index * merged_dim;

                for (0..merge_size) |inner_y| {
                    for (0..merge_size) |inner_x| {
                        const patch_y = group_y * merge_size + inner_y;
                        const patch_x = group_x * merge_size + inner_x;
                        const patch_index = (temporal_group * embeddings.patch_height + patch_y) * embeddings.patch_width + patch_x;
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
    }

    return output;
}

pub fn applyVisualMerger(
    allocator: std.mem.Allocator,
    grouped: MergedPatchGroups,
    weights: VisualMergerWeights,
) !VisualTokens {
    try validateVisualMergerInputs(grouped, weights);

    var hidden = try allocator.alloc(f32, grouped.token_count * weights.fc1.out_features);
    defer allocator.free(hidden);

    var output = VisualTokens{
        .allocator = allocator,
        .data = try allocator.alloc(f32, grouped.token_count * weights.fc2.out_features),
        .token_count = grouped.token_count,
        .embedding_dim = weights.fc2.out_features,
        .grid_time = grouped.temporal_patch_count,
        .grid_width = grouped.merged_width,
        .grid_height = grouped.merged_height,
    };
    errdefer output.deinit();

    for (0..grouped.token_count) |token_index| {
        const input_row = grouped.data[token_index * grouped.merged_dim ..][0..grouped.merged_dim];
        const hidden_row = hidden[token_index * weights.fc1.out_features ..][0..weights.fc1.out_features];
        applyLinear(hidden_row, input_row, weights.fc1);
        cpu.geluInPlace(hidden_row);

        const output_row = output.data[token_index * weights.fc2.out_features ..][0..weights.fc2.out_features];
        applyLinear(output_row, hidden_row, weights.fc2);
    }

    return output;
}

pub fn applyVisionBlockMlp(
    allocator: std.mem.Allocator,
    input: PatchEmbeddings,
    weights: LinearMlpWeights,
    eps: f32,
) !PatchEmbeddings {
    try validateVisionBlockMlpInputs(input, weights);

    const normed = try allocator.alloc(f32, input.embedding_dim);
    defer allocator.free(normed);
    const hidden = try allocator.alloc(f32, weights.fc1.out_features);
    defer allocator.free(hidden);

    var output = PatchEmbeddings{
        .allocator = allocator,
        .data = try allocator.alloc(f32, input.data.len),
        .token_count = input.token_count,
        .embedding_dim = input.embedding_dim,
        .temporal_patch_count = input.temporal_patch_count,
        .patch_width = input.patch_width,
        .patch_height = input.patch_height,
    };
    errdefer output.deinit();

    for (0..input.token_count) |token_index| {
        const src = input.data[token_index * input.embedding_dim ..][0..input.embedding_dim];
        const dst = output.data[token_index * input.embedding_dim ..][0..input.embedding_dim];

        try cpu.layerNorm(normed, src, weights.norm.weight, weights.norm.bias, eps);
        applyLinear(hidden, normed, weights.fc1);
        cpu.geluInPlace(hidden);
        applyLinear(dst, hidden, weights.fc2);

        for (dst, src) |*out, residual| {
            out.* += residual;
        }
    }

    return output;
}

pub fn applyVisionBlockAttention(
    allocator: std.mem.Allocator,
    input: PatchEmbeddings,
    weights: VisualAttentionWeights,
    eps: f32,
) !PatchEmbeddings {
    try validateVisionBlockAttentionInputs(input, weights);

    const token_dim = input.embedding_dim;
    const qkv_dim = weights.qkv.out_features;
    const head_dim = token_dim / weights.num_heads;

    const normed = try allocator.alloc(f32, token_dim);
    defer allocator.free(normed);
    var qkv_buffer = try allocator.alloc(f32, input.token_count * qkv_dim);
    defer allocator.free(qkv_buffer);
    const scores = try allocator.alloc(f32, input.token_count);
    defer allocator.free(scores);
    const attended = try allocator.alloc(f32, token_dim);
    defer allocator.free(attended);

    var output = PatchEmbeddings{
        .allocator = allocator,
        .data = try allocator.alloc(f32, input.data.len),
        .token_count = input.token_count,
        .embedding_dim = input.embedding_dim,
        .temporal_patch_count = input.temporal_patch_count,
        .patch_width = input.patch_width,
        .patch_height = input.patch_height,
    };
    errdefer output.deinit();

    for (0..input.token_count) |token_index| {
        const src = input.data[token_index * token_dim ..][0..token_dim];
        const dst = qkv_buffer[token_index * qkv_dim ..][0..qkv_dim];
        try cpu.layerNorm(normed, src, weights.norm.weight, weights.norm.bias, eps);
        applyLinear(dst, normed, weights.qkv);
    }

    for (0..input.token_count) |query_index| {
        @memset(attended, 0.0);

        for (0..weights.num_heads) |head_index| {
            const q_slice = qSlice(qkv_buffer, query_index, token_dim, head_index, head_dim);
            for (0..input.token_count) |key_index| {
                const k_slice = kSlice(qkv_buffer, key_index, token_dim, head_index, head_dim);
                scores[key_index] = (try cpu.dot(q_slice, k_slice)) / @sqrt(@as(f32, @floatFromInt(head_dim)));
            }
            try attention.softmaxInPlace(scores);

            const out_head = attended[head_index * head_dim ..][0..head_dim];
            for (0..input.token_count) |value_index| {
                const v_slice = vSlice(qkv_buffer, value_index, token_dim, head_index, head_dim);
                try cpu.axpyInPlace(out_head, scores[value_index], v_slice);
            }
        }

        const projected = output.data[query_index * token_dim ..][0..token_dim];
        applyLinear(projected, attended, weights.proj);
        const residual = input.data[query_index * token_dim ..][0..token_dim];
        for (projected, residual) |*out, value| {
            out.* += value;
        }
    }

    return output;
}

pub fn applyPositionEmbeddings(
    allocator: std.mem.Allocator,
    input: PatchEmbeddings,
    weights: PositionEmbeddingWeights,
) !PatchEmbeddings {
    if (input.embedding_dim != weights.embedding_dim) return error.ShapeMismatch;
    if (weights.base_width == 0 or weights.base_height == 0) return error.ShapeMismatch;
    if (weights.data.len != weights.tokenCount() * weights.embedding_dim) return error.ShapeMismatch;

    var output = PatchEmbeddings{
        .allocator = allocator,
        .data = try allocator.alloc(f32, input.data.len),
        .token_count = input.token_count,
        .embedding_dim = input.embedding_dim,
        .temporal_patch_count = input.temporal_patch_count,
        .patch_width = input.patch_width,
        .patch_height = input.patch_height,
    };
    errdefer output.deinit();
    @memcpy(output.data, input.data);

    for (0..input.temporal_patch_count) |temporal_group| {
        for (0..input.patch_height) |y| {
            for (0..input.patch_width) |x| {
                const token_index = (temporal_group * input.patch_height + y) * input.patch_width + x;
                const dst = output.data[token_index * input.embedding_dim ..][0..input.embedding_dim];
                if (input.patch_width == weights.base_width and input.patch_height == weights.base_height) {
                    const spatial_index = y * input.patch_width + x;
                    const src = weights.data[spatial_index * input.embedding_dim ..][0..input.embedding_dim];
                    for (dst, src) |*out, pos| out.* += pos;
                    continue;
                }

                const src_y = scaleCoord(y, input.patch_height, weights.base_height);
                const y0 = clampFloor(src_y, weights.base_height);
                const y1 = @min(y0 + 1, weights.base_height - 1);
                const wy = src_y - @as(f32, @floatFromInt(y0));

                const src_x = scaleCoord(x, input.patch_width, weights.base_width);
                const x0 = clampFloor(src_x, weights.base_width);
                const x1 = @min(x0 + 1, weights.base_width - 1);
                const wx = src_x - @as(f32, @floatFromInt(x0));

                const p00 = weights.data[(y0 * weights.base_width + x0) * input.embedding_dim ..][0..input.embedding_dim];
                const p01 = weights.data[(y0 * weights.base_width + x1) * input.embedding_dim ..][0..input.embedding_dim];
                const p10 = weights.data[(y1 * weights.base_width + x0) * input.embedding_dim ..][0..input.embedding_dim];
                const p11 = weights.data[(y1 * weights.base_width + x1) * input.embedding_dim ..][0..input.embedding_dim];

                for (0..input.embedding_dim) |dim| {
                    const top = lerp(p00[dim], p01[dim], wx);
                    const bottom = lerp(p10[dim], p11[dim], wx);
                    dst[dim] += lerp(top, bottom, wy);
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
    if (input.tensor.batch < weights.temporal_patch_size) return error.ShapeMismatch;
    if (input.grid.patchSize() != weights.patch_size) return error.ShapeMismatch;
    if (input.grid.resized_width != input.tensor.width or input.grid.resized_height != input.tensor.height) return error.ShapeMismatch;
}

fn validateVisualMergerInputs(grouped: MergedPatchGroups, weights: VisualMergerWeights) !void {
    if (weights.fc1.data.len != weights.fc1.expectedLen()) return error.ShapeMismatch;
    if (weights.fc2.data.len != weights.fc2.expectedLen()) return error.ShapeMismatch;
    if (weights.fc1.in_features != grouped.merged_dim) return error.ShapeMismatch;
    if (weights.fc2.in_features != weights.fc1.out_features) return error.ShapeMismatch;
    if (weights.fc1.out_features == 0 or weights.fc2.out_features == 0) return error.ShapeMismatch;
    if (weights.fc1.bias) |bias| {
        if (bias.len != weights.fc1.out_features) return error.ShapeMismatch;
    }
    if (weights.fc2.bias) |bias| {
        if (bias.len != weights.fc2.out_features) return error.ShapeMismatch;
    }
}

fn validateVisionBlockMlpInputs(input: PatchEmbeddings, weights: LinearMlpWeights) !void {
    if (weights.norm.dim != input.embedding_dim) return error.ShapeMismatch;
    if (weights.norm.weight.len != input.embedding_dim or weights.norm.bias.len != input.embedding_dim) return error.ShapeMismatch;
    if (weights.fc1.data.len != weights.fc1.expectedLen()) return error.ShapeMismatch;
    if (weights.fc2.data.len != weights.fc2.expectedLen()) return error.ShapeMismatch;
    if (weights.fc1.in_features != input.embedding_dim) return error.ShapeMismatch;
    if (weights.fc2.in_features != weights.fc1.out_features) return error.ShapeMismatch;
    if (weights.fc2.out_features != input.embedding_dim) return error.ShapeMismatch;
    if (weights.fc1.bias) |bias| {
        if (bias.len != weights.fc1.out_features) return error.ShapeMismatch;
    }
    if (weights.fc2.bias) |bias| {
        if (bias.len != weights.fc2.out_features) return error.ShapeMismatch;
    }
}

fn validateVisionBlockAttentionInputs(input: PatchEmbeddings, weights: VisualAttentionWeights) !void {
    if (weights.norm.dim != input.embedding_dim) return error.ShapeMismatch;
    if (weights.norm.weight.len != input.embedding_dim or weights.norm.bias.len != input.embedding_dim) return error.ShapeMismatch;
    if (weights.num_heads == 0 or input.embedding_dim % weights.num_heads != 0) return error.ShapeMismatch;
    if (weights.qkv.data.len != weights.qkv.expectedLen()) return error.ShapeMismatch;
    if (weights.proj.data.len != weights.proj.expectedLen()) return error.ShapeMismatch;
    if (weights.qkv.in_features != input.embedding_dim) return error.ShapeMismatch;
    if (weights.qkv.out_features != input.embedding_dim * 3) return error.ShapeMismatch;
    if (weights.proj.in_features != input.embedding_dim or weights.proj.out_features != input.embedding_dim) return error.ShapeMismatch;
    if (weights.qkv.bias) |bias| {
        if (bias.len != weights.qkv.out_features) return error.ShapeMismatch;
    }
    if (weights.proj.bias) |bias| {
        if (bias.len != weights.proj.out_features) return error.ShapeMismatch;
    }
}

fn applyLinear(output: []f32, input: []const f32, weights: LinearWeights) void {
    for (0..weights.out_features) |row| {
        var sum = if (weights.bias) |bias| bias[row] else 0.0;
        const row_base = row * weights.in_features;
        for (0..weights.in_features) |col| {
            sum += weights.data[row_base + col] * input[col];
        }
        output[row] = sum;
    }
}

fn qSlice(qkv: []const f32, token_index: usize, dim: usize, head_index: usize, head_dim: usize) []const f32 {
    const base = token_index * dim * 3 + head_index * head_dim;
    return qkv[base .. base + head_dim];
}

fn kSlice(qkv: []const f32, token_index: usize, dim: usize, head_index: usize, head_dim: usize) []const f32 {
    const base = token_index * dim * 3 + dim + head_index * head_dim;
    return qkv[base .. base + head_dim];
}

fn vSlice(qkv: []const f32, token_index: usize, dim: usize, head_index: usize, head_dim: usize) []const f32 {
    const base = token_index * dim * 3 + dim * 2 + head_index * head_dim;
    return qkv[base .. base + head_dim];
}

fn scaleCoord(index: usize, dst_size: usize, src_size: usize) f32 {
    if (dst_size <= 1 or src_size <= 1) return 0.0;
    return @as(f32, @floatFromInt(index)) * @as(f32, @floatFromInt(src_size - 1)) / @as(f32, @floatFromInt(dst_size - 1));
}

fn clampFloor(value: f32, upper: usize) usize {
    if (value <= 0.0) return 0;
    const idx = @as(usize, @intFromFloat(@floor(value)));
    return @min(idx, upper - 1);
}

fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn tensorIndex(tensor: imaging.TensorF32NCHW, batch: usize, channel: usize, y: usize, x: usize) usize {
    return batch * tensor.stride_n + channel * tensor.stride_c + y * tensor.stride_h + x;
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
        .temporal_patch_count = 1,
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

test "chandra visual merger projects grouped patches into visual tokens" {
    var grouped = MergedPatchGroups{
        .allocator = std.testing.allocator,
        .data = try std.testing.allocator.dupe(f32, &.{ 1, 2, 3, 4 }),
        .token_count = 1,
        .merged_dim = 4,
        .merged_width = 1,
        .merged_height = 1,
    };
    defer grouped.deinit();

    const fc1 = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const fc1_bias = [_]f32{ 0, 1, -1 };
    const fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const fc2_bias = [_]f32{ 0.5, -0.5 };

    var tokens = try applyVisualMerger(std.testing.allocator, grouped, .{
        .fc1 = .{
            .data = &fc1,
            .bias = &fc1_bias,
            .out_features = 3,
            .in_features = 4,
        },
        .fc2 = .{
            .data = &fc2,
            .bias = &fc2_bias,
            .out_features = 2,
            .in_features = 3,
        },
    });
    defer tokens.deinit();

    try std.testing.expectEqual(@as(usize, 1), tokens.token_count);
    try std.testing.expectEqual(@as(usize, 2), tokens.embedding_dim);
    try std.testing.expectEqual(@as(usize, 1), tokens.grid_time);
    try std.testing.expectEqual(@as(usize, 1), tokens.grid_width);
    try std.testing.expectEqual(@as(usize, 1), tokens.grid_height);
    try std.testing.expectApproxEqAbs(@as(f32, 1.341192), tokens.at(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.295359), tokens.at(0, 1), 0.0001);
}

test "chandra vision block mlp applies norm gelu mlp with residual" {
    var embeddings = PatchEmbeddings{
        .allocator = std.testing.allocator,
        .data = try std.testing.allocator.dupe(f32, &.{
            1, 2,
            3, 4,
        }),
        .token_count = 2,
        .embedding_dim = 2,
        .temporal_patch_count = 1,
        .patch_width = 2,
        .patch_height = 1,
    };
    defer embeddings.deinit();

    const norm_weight = [_]f32{ 1, 1 };
    const norm_bias = [_]f32{ 0, 0 };
    const fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const fc1_bias = [_]f32{ 0, 0, 0 };
    const fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const fc2_bias = [_]f32{ 0, 0 };

    var output = try applyVisionBlockMlp(std.testing.allocator, embeddings, .{
        .norm = .{
            .weight = &norm_weight,
            .bias = &norm_bias,
            .dim = 2,
        },
        .fc1 = .{
            .data = &fc1,
            .bias = &fc1_bias,
            .out_features = 3,
            .in_features = 2,
        },
        .fc2 = .{
            .data = &fc2,
            .bias = &fc2_bias,
            .out_features = 2,
            .in_features = 3,
        },
    }, 1e-5);
    defer output.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 0.158811), output.at(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.317622), output.at(0, 1), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.158811), output.at(1, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.317622), output.at(1, 1), 0.0001);
}

test "chandra vision block attention applies qkv proj attention with residual" {
    var embeddings = PatchEmbeddings{
        .allocator = std.testing.allocator,
        .data = try std.testing.allocator.dupe(f32, &.{
            1,  -1,
            -1, 1,
        }),
        .token_count = 2,
        .embedding_dim = 2,
        .temporal_patch_count = 1,
        .patch_width = 2,
        .patch_height = 1,
    };
    defer embeddings.deinit();

    const norm_weight = [_]f32{ 1, 1 };
    const norm_bias = [_]f32{ 0, 0 };
    const qkv = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    const qkv_bias = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const proj = [_]f32{
        1, 0,
        0, 1,
    };
    const proj_bias = [_]f32{ 0, 0 };

    var output = try applyVisionBlockAttention(std.testing.allocator, embeddings, .{
        .norm = .{
            .weight = &norm_weight,
            .bias = &norm_bias,
            .dim = 2,
        },
        .qkv = .{
            .data = &qkv,
            .bias = &qkv_bias,
            .out_features = 6,
            .in_features = 2,
        },
        .proj = .{
            .data = &proj,
            .bias = &proj_bias,
            .out_features = 2,
            .in_features = 2,
        },
        .num_heads = 1,
    }, 1e-5);
    defer output.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.888386), output.at(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.888386), output.at(0, 1), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.888386), output.at(1, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.888386), output.at(1, 1), 0.0001);
}

test "chandra position embeddings add resized learned grid embeddings" {
    var embeddings = PatchEmbeddings{
        .allocator = std.testing.allocator,
        .data = try std.testing.allocator.dupe(f32, &.{
            0, 0,
            0, 0,
        }),
        .token_count = 2,
        .embedding_dim = 2,
        .temporal_patch_count = 1,
        .patch_width = 2,
        .patch_height = 1,
    };
    defer embeddings.deinit();

    const pos = [_]f32{
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    };

    var output = try applyPositionEmbeddings(std.testing.allocator, embeddings, .{
        .data = &pos,
        .base_width = 2,
        .base_height = 2,
        .embedding_dim = 2,
    });
    defer output.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1), output.at(0, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 10), output.at(0, 1), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2), output.at(1, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 20), output.at(1, 1), 0.0001);
}

test "chandra patch embedding consumes temporal frame groups" {
    var image_a = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image_a.deinit();
    image_a.fill(1);

    var image_b = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image_b.deinit();
    image_b.fill(2);

    const frames = [_]*const imaging.ImageU8{ &image_a, &image_b };
    var prepared = try preprocess.prepareImageFramesInput(std.testing.allocator, &frames, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 2,
        .temporal_patch_size = 2,
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
        3, 3, 3, 3,
        0, 0, 0, 0,
        0, 0, 0, 0,
        4, 4, 4, 4,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };

    var embeddings = try patchEmbedImage(std.testing.allocator, &prepared, .{
        .data = &weights,
        .out_channels = 2,
        .in_channels = 3,
        .temporal_patch_size = 2,
        .patch_size = 2,
    });
    defer embeddings.deinit();

    try std.testing.expectEqual(@as(usize, 1), embeddings.temporal_patch_count);
    try std.testing.expectEqual(@as(usize, 1), embeddings.token_count);
    try std.testing.expectEqual(@as(f32, 36), embeddings.at(0, 0));
    try std.testing.expectEqual(@as(f32, 84), embeddings.at(0, 1));
}
