const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const catalog_mod = @import("../catalog/catalog.zig");
const normalized = @import("../model/resolver/normalized_model.zig");
const types = @import("../types.zig");

const bert = @import("text/bert/family.zig");
const qwen3 = @import("text/qwen3/family.zig");
const yolo = @import("vision/yolo/family.zig");
const swiftocr = @import("ocr/swiftocr/family.zig");
const chandra = @import("ocr/chandra/family.zig");
const moss_tts_nano = @import("tts/moss_tts_nano/family.zig");
const generic = @import("generic/family.zig");

pub const NormalizeFn = *const fn (
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) anyerror!?normalized.NormalizedModel;

pub const ProviderDescriptor = struct {
    key: types.ProviderKey,
    modality: types.Modality,
    family: []const u8,
};

pub const FamilyEntry = struct {
    descriptor: ProviderDescriptor,
    backend: *const backend_mod.RuntimeBackend,
    try_normalize: NormalizeFn,
};

const builtin_families = [_]FamilyEntry{
    .{
        .descriptor = .{ .key = qwen3.key, .modality = qwen3.modality, .family = qwen3.family_name },
        .backend = &qwen3.backend,
        .try_normalize = qwen3.tryNormalize,
    },
    .{
        .descriptor = .{ .key = bert.key, .modality = bert.modality, .family = bert.family_name },
        .backend = &bert.backend,
        .try_normalize = bert.tryNormalize,
    },
    .{
        .descriptor = .{ .key = yolo.key, .modality = yolo.modality, .family = yolo.family_name },
        .backend = &yolo.backend,
        .try_normalize = yolo.tryNormalize,
    },
    .{
        .descriptor = .{ .key = swiftocr.key, .modality = swiftocr.modality, .family = swiftocr.family_name },
        .backend = &swiftocr.backend,
        .try_normalize = swiftocr.tryNormalize,
    },
    .{
        .descriptor = .{ .key = chandra.key, .modality = chandra.modality, .family = chandra.family_name },
        .backend = &chandra.backend,
        .try_normalize = chandra.tryNormalize,
    },
    .{
        .descriptor = .{ .key = moss_tts_nano.key, .modality = moss_tts_nano.modality, .family = moss_tts_nano.family_name },
        .backend = &moss_tts_nano.backend,
        .try_normalize = moss_tts_nano.tryNormalize,
    },
    .{
        .descriptor = .{ .key = generic.key, .modality = generic.modality, .family = generic.family_name },
        .backend = &generic.backend,
        .try_normalize = generic.tryNormalize,
    },
};

pub fn builtinFamilies() []const FamilyEntry {
    return &builtin_families;
}

pub fn findBuiltinByKey(key: types.ProviderKey) ?FamilyEntry {
    for (builtin_families) |entry| {
        if (entry.descriptor.key == key) return entry;
    }
    return null;
}

test "family registry exposes builtin family entries" {
    try std.testing.expectEqual(@as(usize, 7), builtinFamilies().len);
    const qwen = findBuiltinByKey(.qwen3_text) orelse return error.ExpectedFamily;
    try std.testing.expectEqualStrings("qwen3", qwen.descriptor.family);
    try std.testing.expect(qwen.backend.provider_key == .qwen3_text);
}
