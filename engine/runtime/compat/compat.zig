const std = @import("std");
const catalog_mod = @import("../catalog/catalog.zig");
const normalized = @import("normalized_model.zig");
const qwen3 = @import("providers/qwen3.zig");
const bert = @import("providers/bert.zig");
const yolo = @import("providers/yolo.zig");
const swiftocr = @import("providers/swiftocr.zig");
const generic = @import("providers/generic.zig");
const types = @import("../types.zig");

pub const CompatibilityReport = @import("capability_report.zig").CompatibilityReport;
pub const NormalizedModel = normalized.NormalizedModel;
pub const RuntimeArtifactSet = normalized.RuntimeArtifactSet;
pub const RuntimeCapabilitySet = normalized.RuntimeCapabilitySet;
pub const RuntimeModelDescriptor = normalized.RuntimeModelDescriptor;

pub fn normalizeModel(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: types.WeightScheme,
) !NormalizedModel {
    var catalog = try catalog_mod.ArtifactCatalog.discover(allocator, model_dir);
    defer catalog.deinit();
    return try normalizeCatalog(allocator, &catalog, preferred_weights);
}

pub fn normalizeCatalog(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !NormalizedModel {
    if (try qwen3.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try bert.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try yolo.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try swiftocr.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try generic.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    return error.UnsupportedModelDirectory;
}

test "compat normalizes qwen3 model directories into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.qwen3_text, model.provider_key);
    try std.testing.expectEqual(types.Modality.text, model.descriptor.modality);
    try std.testing.expectEqualStrings("qwen3", model.descriptor.family);
    try std.testing.expect(model.capabilities.supports_stream);
}

test "compat normalizes yolo graph directories into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json", "{}");
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.yolo_vision, model.provider_key);
    try std.testing.expectEqual(types.Modality.vision, model.descriptor.modality);
    try std.testing.expectEqual(types.CompatibilityStatus.degraded, model.compat.status);
}

test "compat normalizes swiftocr bundles into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "demo.swm", "SWOCR01");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.swiftocr_ocr, model.provider_key);
    try std.testing.expectEqual(types.Modality.ocr, model.descriptor.modality);
    try std.testing.expectEqual(types.CompatibilityStatus.degraded, model.compat.status);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
