const std = @import("std");
const safetensors = @import("../text/safetensors.zig");
const fs_compat = @import("engine_fs_compat");

pub const TensorGroup = enum {
    text,
    vision,
    projector,
    output,
    other,
};

pub const TensorRecord = struct {
    name: []u8,
    relative_file: []u8,
    group: TensorGroup,
    dtype: ?safetensors.DType = null,
    shape: []u64 = &.{},
};

pub const GroupCounts = struct {
    text: usize = 0,
    vision: usize = 0,
    projector: usize = 0,
    output: usize = 0,
    other: usize = 0,

    pub fn add(self: *GroupCounts, group: TensorGroup) void {
        switch (group) {
            .text => self.text += 1,
            .vision => self.vision += 1,
            .projector => self.projector += 1,
            .output => self.output += 1,
            .other => self.other += 1,
        }
    }
};

pub const TensorManifest = struct {
    allocator: std.mem.Allocator,
    tensors: []TensorRecord,
    counts: GroupCounts,
    source: enum { safetensors_file, safetensors_index },

    pub fn deinit(self: *TensorManifest) void {
        for (self.tensors) |tensor| {
            self.allocator.free(tensor.name);
            self.allocator.free(tensor.relative_file);
            if (tensor.shape.len != 0) self.allocator.free(tensor.shape);
        }
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    pub fn len(self: TensorManifest) usize {
        return self.tensors.len;
    }

    pub fn findPatchEmbeddingWeight(self: TensorManifest) ?*const TensorRecord {
        for (self.tensors) |*tensor| {
            if (isPatchEmbeddingWeight(tensor.name)) return tensor;
        }
        return null;
    }
};

pub fn loadManifest(allocator: std.mem.Allocator, model_path: []const u8) !TensorManifest {
    if (try findIndexPath(allocator, model_path)) |index_path| {
        defer allocator.free(index_path);
        return try loadFromIndex(allocator, index_path);
    }

    return try loadFromSafetensorsFiles(allocator, model_path);
}

fn loadFromIndex(allocator: std.mem.Allocator, index_path: []const u8) !TensorManifest {
    const bytes = try fs_compat.cwd().readFileAlloc(allocator, index_path, 32 * 1024 * 1024);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidSafetensorsIndex;
    const weight_map = parsed.value.object.get("weight_map") orelse return error.MissingWeightMap;
    if (weight_map != .object) return error.InvalidWeightMap;

    var records = std.ArrayListUnmanaged(TensorRecord).empty;
    errdefer deinitRecordList(allocator, records.items);

    var counts: GroupCounts = .{};
    var it = weight_map.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string) return error.InvalidWeightMapEntry;

        const group = classifyTensor(entry.key_ptr.*);
        counts.add(group);
        try records.append(allocator, .{
            .name = try allocator.dupe(u8, entry.key_ptr.*),
            .relative_file = try allocator.dupe(u8, entry.value_ptr.*.string),
            .group = group,
        });
    }

    return .{
        .allocator = allocator,
        .tensors = try records.toOwnedSlice(allocator),
        .counts = counts,
        .source = .safetensors_index,
    };
}

fn loadFromSafetensorsFiles(allocator: std.mem.Allocator, model_path: []const u8) !TensorManifest {
    var dir = try fs_compat.openDirAbsolute(model_path, .{ .iterate = true });
    defer dir.close();

    var records = std.ArrayListUnmanaged(TensorRecord).empty;
    errdefer deinitRecordList(allocator, records.items);

    var counts: GroupCounts = .{};
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;

        const absolute_path = try std.fs.path.join(allocator, &.{ model_path, entry.name });
        defer allocator.free(absolute_path);

        var parsed_file = try safetensors.loadFromFile(allocator, absolute_path);
        defer parsed_file.deinit();

        var tensor_iter = parsed_file.tensors.iterator();
        while (tensor_iter.next()) |tensor_entry| {
            const group = classifyTensor(tensor_entry.key_ptr.*);
            counts.add(group);
            try records.append(allocator, .{
                .name = try allocator.dupe(u8, tensor_entry.key_ptr.*),
                .relative_file = try allocator.dupe(u8, entry.name),
                .group = group,
                .dtype = tensor_entry.value_ptr.dtype,
                .shape = try allocator.dupe(u64, tensor_entry.value_ptr.shape),
            });
        }
    }

    if (records.items.len == 0) return error.NoSafetensorsWeightsFound;

    return .{
        .allocator = allocator,
        .tensors = try records.toOwnedSlice(allocator),
        .counts = counts,
        .source = .safetensors_file,
    };
}

pub fn classifyTensor(name: []const u8) TensorGroup {
    if (isVisionTensor(name)) return .vision;
    if (isProjectorTensor(name)) return .projector;
    if (isOutputTensor(name)) return .output;
    if (isTextTensor(name)) return .text;
    return .other;
}

pub fn isPatchEmbeddingWeight(name: []const u8) bool {
    return (std.mem.indexOf(u8, name, "patch_embed") != null or
        std.mem.indexOf(u8, name, "patch_embedding") != null) and
        std.mem.endsWith(u8, name, ".weight");
}

fn isVisionTensor(name: []const u8) bool {
    return startsOrContains(name, "visual.") or
        startsOrContains(name, "vision_model.") or
        startsOrContains(name, "vision_tower.") or
        std.mem.indexOf(u8, name, ".visual.") != null or
        std.mem.indexOf(u8, name, ".vision_model.") != null or
        std.mem.indexOf(u8, name, ".vision_tower.") != null or
        std.mem.indexOf(u8, name, "patch_embed") != null;
}

fn isProjectorTensor(name: []const u8) bool {
    return startsOrContains(name, "multi_modal_projector.") or
        startsOrContains(name, "mm_projector.") or
        std.mem.indexOf(u8, name, ".multi_modal_projector.") != null or
        std.mem.indexOf(u8, name, ".mm_projector.") != null or
        std.mem.indexOf(u8, name, ".merger.") != null or
        std.mem.indexOf(u8, name, "visual.merger.") != null;
}

fn isOutputTensor(name: []const u8) bool {
    return std.mem.eql(u8, name, "lm_head.weight") or
        std.mem.endsWith(u8, name, ".lm_head.weight");
}

fn isTextTensor(name: []const u8) bool {
    return startsOrContains(name, "model.") or
        startsOrContains(name, "language_model.") or
        startsOrContains(name, "text_model.") or
        std.mem.indexOf(u8, name, ".language_model.") != null or
        std.mem.indexOf(u8, name, ".text_model.") != null;
}

fn startsOrContains(name: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, name, prefix);
}

fn findIndexPath(allocator: std.mem.Allocator, model_path: []const u8) !?[]u8 {
    const fixed_path = try std.fs.path.join(allocator, &.{ model_path, "model.safetensors.index.json" });
    if (pathExists(fixed_path)) return fixed_path;
    allocator.free(fixed_path);

    var dir = try fs_compat.openDirAbsolute(model_path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors.index.json")) continue;
        return try std.fs.path.join(allocator, &.{ model_path, entry.name });
    }

    return null;
}

fn pathExists(path: []const u8) bool {
    const file = fs_compat.openFileAbsolute(path, .{}) catch return false;
    file.close();
    return true;
}

fn deinitRecordList(allocator: std.mem.Allocator, records: []TensorRecord) void {
    for (records) |record| {
        allocator.free(record.name);
        allocator.free(record.relative_file);
        if (record.shape.len != 0) allocator.free(record.shape);
    }
}

test "chandra weight manifest parses safetensors index weight map" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "model.safetensors.index.json",
        \\{
        \\  "metadata": {"total_size": 64},
        \\  "weight_map": {
        \\    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        \\    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
        \\    "visual.patch_embed.proj.weight": "model-00002-of-00002.safetensors",
        \\    "visual.merger.mlp.0.weight": "model-00002-of-00002.safetensors",
        \\    "lm_head.weight": "model-00002-of-00002.safetensors"
        \\  }
        \\}
    );

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var manifest = try loadManifest(std.testing.allocator, root_path);
    defer manifest.deinit();

    try std.testing.expectEqual(@as(usize, 5), manifest.len());
    try std.testing.expectEqual(@as(usize, 2), manifest.counts.text);
    try std.testing.expectEqual(@as(usize, 1), manifest.counts.vision);
    try std.testing.expectEqual(@as(usize, 1), manifest.counts.projector);
    try std.testing.expectEqual(@as(usize, 1), manifest.counts.output);
    try std.testing.expect(manifest.findPatchEmbeddingWeight() != null);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
