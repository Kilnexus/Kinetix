const std = @import("std");
const tensor_store = @import("../text/storage/store.zig");
const chandra_vision = @import("chandra_vision.zig");
const chandra_weights = @import("chandra_weights.zig");

pub const FileStore = struct {
    relative_path: []u8,
    store: tensor_store.TensorStore,
};

pub const ChandraStore = struct {
    allocator: std.mem.Allocator,
    files: []FileStore,

    pub fn open(allocator: std.mem.Allocator, model_path: []const u8) !ChandraStore {
        var manifest = try chandra_weights.loadManifest(allocator, model_path);
        defer manifest.deinit();

        var files = std.ArrayListUnmanaged(FileStore).empty;
        errdefer deinitFiles(allocator, files.items);

        for (manifest.tensors) |tensor| {
            if (containsFile(files.items, tensor.relative_file)) continue;

            const absolute_path = try std.fs.path.join(allocator, &.{ model_path, tensor.relative_file });
            defer allocator.free(absolute_path);

            try files.append(allocator, .{
                .relative_path = try allocator.dupe(u8, tensor.relative_file),
                .store = try tensor_store.TensorStore.open(allocator, absolute_path),
            });
        }

        return .{
            .allocator = allocator,
            .files = try files.toOwnedSlice(allocator),
        };
    }

    pub fn deinit(self: *ChandraStore) void {
        deinitFiles(self.allocator, self.files);
        self.* = undefined;
    }

    pub fn loadPatchEmbeddingWeights(
        self: *const ChandraStore,
        allocator: std.mem.Allocator,
    ) !chandra_vision.OwnedPatchEmbeddingWeights {
        const weight_name = self.findPatchEmbeddingWeightName() orelse return error.TensorNotFound;
        const weight_store = self.findStoreForTensor(weight_name) orelse return error.TensorNotFound;
        const weight_info = weight_store.getTensor(weight_name).?;

        const dims = try patchEmbeddingShape(weight_info.shape);
        const weights = try weight_store.readElementsAsF32Alloc(weight_name, 0, @intCast(try weight_info.elementCount()));
        errdefer allocator.free(weights);

        var bias: ?[]f32 = null;
        errdefer if (bias) |value| allocator.free(value);
        const bias_name = try patchEmbeddingBiasName(allocator, weight_name);
        defer allocator.free(bias_name);
        if (self.findStoreForTensor(bias_name)) |bias_store| {
            if (bias_store.getTensor(bias_name)) |bias_info| {
                bias = try bias_store.readElementsAsF32Alloc(bias_name, 0, @intCast(try bias_info.elementCount()));
            }
        }

        return .{
            .allocator = allocator,
            .weights = .{
                .data = weights,
                .bias = bias,
                .out_channels = dims.out_channels,
                .in_channels = dims.in_channels,
                .temporal_patch_size = dims.temporal_patch_size,
                .patch_size = dims.patch_size,
            },
        };
    }

    pub fn loadVisualMergerWeights(
        self: *const ChandraStore,
        allocator: std.mem.Allocator,
    ) !chandra_vision.OwnedVisualMergerWeights {
        const fc1 = try self.loadLinearWeights(allocator, &.{ "visual.merger.mlp.0.weight", "merger.mlp.0.weight" });
        errdefer fc1.deinit();
        const fc2 = try self.loadLinearWeights(allocator, &.{ "visual.merger.mlp.2.weight", "merger.mlp.2.weight" });
        errdefer fc2.deinit();

        return .{
            .allocator = allocator,
            .weights = .{
                .fc1 = fc1.weights,
                .fc2 = fc2.weights,
            },
        };
    }

    fn findPatchEmbeddingWeightName(self: *const ChandraStore) ?[]const u8 {
        for (self.files) |file| {
            var it = file.store.parsed.tensors.iterator();
            while (it.next()) |entry| {
                if (chandra_weights.isPatchEmbeddingWeight(entry.key_ptr.*)) return entry.key_ptr.*;
            }
        }
        return null;
    }

    fn findStoreForTensor(self: *const ChandraStore, name: []const u8) ?*const tensor_store.TensorStore {
        for (self.files) |*file| {
            if (file.store.getTensor(name) != null) return &file.store;
        }
        return null;
    }

    fn loadLinearWeights(
        self: *const ChandraStore,
        allocator: std.mem.Allocator,
        candidate_names: []const []const u8,
    ) !OwnedLinearWeights {
        const weight_name = self.findFirstTensorName(candidate_names) orelse return error.TensorNotFound;
        const weight_store = self.findStoreForTensor(weight_name) orelse return error.TensorNotFound;
        const weight_info = weight_store.getTensor(weight_name).?;
        if (weight_info.rank() != 2) return error.InvalidTensorRank;

        const out_features = dimToUsize(weight_info.shape[0]);
        const in_features = dimToUsize(weight_info.shape[1]);
        const data = try weight_store.readElementsAsF32Alloc(weight_name, 0, @intCast(try weight_info.elementCount()));
        errdefer allocator.free(data);

        var bias: ?[]f32 = null;
        errdefer if (bias) |value| allocator.free(value);
        const bias_name = try biasTensorName(allocator, weight_name);
        defer allocator.free(bias_name);
        if (self.findStoreForTensor(bias_name)) |bias_store| {
            if (bias_store.getTensor(bias_name)) |bias_info| {
                bias = try bias_store.readElementsAsF32Alloc(bias_name, 0, @intCast(try bias_info.elementCount()));
            }
        }

        return .{
            .allocator = allocator,
            .weights = .{
                .data = data,
                .bias = bias,
                .out_features = out_features,
                .in_features = in_features,
            },
        };
    }

    fn findFirstTensorName(self: *const ChandraStore, candidates: []const []const u8) ?[]const u8 {
        for (candidates) |name| {
            if (self.findStoreForTensor(name) != null) return name;
        }
        return null;
    }
};

fn containsFile(items: []const FileStore, relative_path: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item.relative_path, relative_path)) return true;
    }
    return false;
}

fn deinitFiles(allocator: std.mem.Allocator, items: []FileStore) void {
    for (items) |*item| {
        item.store.deinit();
        allocator.free(item.relative_path);
    }
    allocator.free(items);
}

const PatchEmbeddingShape = struct {
    out_channels: usize,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
};

const OwnedLinearWeights = struct {
    allocator: std.mem.Allocator,
    weights: chandra_vision.LinearWeights,

    fn deinit(self: OwnedLinearWeights) void {
        self.allocator.free(self.weights.data);
        if (self.weights.bias) |bias| self.allocator.free(bias);
    }
};

fn patchEmbeddingShape(shape: []const u64) !PatchEmbeddingShape {
    return switch (shape.len) {
        5 => .{
            .out_channels = dimToUsize(shape[0]),
            .in_channels = dimToUsize(shape[1]),
            .temporal_patch_size = dimToUsize(shape[2]),
            .patch_size = blk: {
                if (shape[3] != shape[4]) return error.ShapeMismatch;
                break :blk dimToUsize(shape[3]);
            },
        },
        4 => .{
            .out_channels = dimToUsize(shape[0]),
            .in_channels = dimToUsize(shape[1]),
            .temporal_patch_size = 1,
            .patch_size = blk: {
                if (shape[2] != shape[3]) return error.ShapeMismatch;
                break :blk dimToUsize(shape[2]);
            },
        },
        else => error.InvalidTensorRank,
    };
}

fn patchEmbeddingBiasName(allocator: std.mem.Allocator, weight_name: []const u8) ![]u8 {
    return biasTensorName(allocator, weight_name);
}

fn biasTensorName(allocator: std.mem.Allocator, weight_name: []const u8) ![]u8 {
    if (!std.mem.endsWith(u8, weight_name, ".weight")) return error.InvalidTensorName;
    const stem = weight_name[0 .. weight_name.len - ".weight".len];
    return std.mem.concat(allocator, u8, &.{ stem, ".bias" });
}

fn dimToUsize(value: u64) usize {
    return @intCast(value);
}

test "chandra store loads patch embedding weights from a synthetic safetensors file" {
    const header =
        \\{"visual.patch_embed.proj.weight":{"dtype":"F32","shape":[2,3,1,2,2],"data_offsets":[0,96]},"visual.patch_embed.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[96,104]}}
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const model_dir_name = "model";
    try tmp.dir.makeDir(model_dir_name);
    var model_dir = try tmp.dir.openDir(model_dir_name, .{});
    defer model_dir.close();

    const file = try model_dir.createFile("model.safetensors", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    var payload: [104]u8 = undefined;
    const weight_values = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    for (weight_values, 0..) |value, index| {
        std.mem.writeInt(u32, payload[index * 4 .. index * 4 + 4][0..4], @bitCast(value), .little);
    }
    std.mem.writeInt(u32, payload[96..100], @bitCast(@as(f32, 0.0)), .little);
    std.mem.writeInt(u32, payload[100..104], @bitCast(@as(f32, 1.0)), .little);
    try file.writeAll(&payload);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, model_dir_name);
    defer std.testing.allocator.free(root_path);

    var store = try ChandraStore.open(std.testing.allocator, root_path);
    defer store.deinit();

    var loaded = try store.loadPatchEmbeddingWeights(std.testing.allocator);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.weights.out_channels);
    try std.testing.expectEqual(@as(usize, 3), loaded.weights.in_channels);
    try std.testing.expectEqual(@as(usize, 1), loaded.weights.temporal_patch_size);
    try std.testing.expectEqual(@as(usize, 2), loaded.weights.patch_size);
    try std.testing.expectEqual(@as(f32, 1.0), loaded.weights.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), loaded.weights.bias.?[1]);
}

test "chandra store loads visual merger weights from a synthetic safetensors file" {
    const header =
        \\{"visual.merger.mlp.0.weight":{"dtype":"F32","shape":[3,4],"data_offsets":[0,48]},"visual.merger.mlp.0.bias":{"dtype":"F32","shape":[3],"data_offsets":[48,60]},"visual.merger.mlp.2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[60,84]},"visual.merger.mlp.2.bias":{"dtype":"F32","shape":[2],"data_offsets":[84,92]}}
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const model_dir_name = "model";
    try tmp.dir.makeDir(model_dir_name);
    var model_dir = try tmp.dir.openDir(model_dir_name, .{});
    defer model_dir.close();

    const file = try model_dir.createFile("model.safetensors", .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header);

    var payload: [92]u8 = undefined;
    const values = [_]f32{
        1, 0,   0,    0,
        0, 1,   0,    0,
        0, 0,   1,    0,
        0, 1,   -1,   1,
        0, 0,   0,    1,
        1, 0.5, -0.5,
    };
    for (values, 0..) |value, index| {
        std.mem.writeInt(u32, payload[index * 4 .. index * 4 + 4][0..4], @bitCast(value), .little);
    }
    try file.writeAll(&payload);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, model_dir_name);
    defer std.testing.allocator.free(root_path);

    var store = try ChandraStore.open(std.testing.allocator, root_path);
    defer store.deinit();

    var loaded = try store.loadVisualMergerWeights(std.testing.allocator);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 3), loaded.weights.fc1.out_features);
    try std.testing.expectEqual(@as(usize, 4), loaded.weights.fc1.in_features);
    try std.testing.expectEqual(@as(usize, 2), loaded.weights.fc2.out_features);
    try std.testing.expectEqual(@as(f32, 1.0), loaded.weights.fc1.data[0]);
    try std.testing.expectEqual(@as(f32, -0.5), loaded.weights.fc2.bias.?[1]);
}
