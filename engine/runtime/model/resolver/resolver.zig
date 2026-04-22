const std = @import("std");
const catalog_mod = @import("../../catalog/catalog.zig");
const normalized = @import("normalized_model.zig");
const qwen3 = @import("../../families/text/qwen3/family.zig");
const bert = @import("../../families/text/bert/family.zig");
const yolo = @import("../../families/vision/yolo/family.zig");
const swiftocr = @import("../../families/ocr/swiftocr/family.zig");
const chandra = @import("../../families/ocr/chandra/family.zig");
const moss_tts_nano = @import("../../families/tts/moss_tts_nano/family.zig");
const generic = @import("../../families/generic/family.zig");
const types = @import("../../types.zig");

pub const RuntimeSupportReport = @import("support_report.zig").RuntimeSupportReport;
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
    if (try chandra.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try moss_tts_nano.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    if (try generic.tryNormalize(allocator, catalog, preferred_weights)) |model| return model;
    return error.UnsupportedModelDirectory;
}

test "support normalizes qwen3 model directories into a runtime model" {
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

test "support normalizes yolo graph directories into a runtime model" {
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
    try std.testing.expectEqual(types.RuntimeSupportStatus.degraded, model.support.status);
}

test "support normalizes swiftocr bundles into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "demo.swm", "SWOCR01");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.swiftocr_ocr, model.provider_key);
    try std.testing.expectEqual(types.Modality.ocr, model.descriptor.modality);
    try std.testing.expectEqual(types.RuntimeSupportStatus.supported, model.support.status);
}

test "support normalizes chandra huggingface directories into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makeDir("chandra-ocr-2");
    var model_dir = try tmp.dir.openDir("chandra-ocr-2", .{});
    defer model_dir.close();

    try writeTmpFile(model_dir, "config.json", "{\"model_type\":\"qwen3_5\"}");
    try writeTmpFile(model_dir, "tokenizer.json", "{}");
    try writeTmpFile(model_dir, "model.safetensors", "weights");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, "chandra-ocr-2");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.chandra_ocr, model.provider_key);
    try std.testing.expectEqual(types.Modality.ocr, model.descriptor.modality);
    try std.testing.expectEqualStrings("chandra", model.descriptor.family);
    try std.testing.expectEqual(types.RuntimeSupportStatus.supported, model.support.status);
}

test "support normalizes moss tts nano onnx bundle directories into a runtime model" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makeDir("MOSS-TTS-Nano-100M-ONNX");
    try tmp.dir.makeDir("MOSS-Audio-Tokenizer-Nano-ONNX");
    var tts_dir = try tmp.dir.openDir("MOSS-TTS-Nano-100M-ONNX", .{});
    defer tts_dir.close();
    var codec_dir = try tmp.dir.openDir("MOSS-Audio-Tokenizer-Nano-ONNX", .{});
    defer codec_dir.close();

    try writeTmpFile(tts_dir, "browser_poc_manifest.json", "{\"builtin_voices\":[],\"model_files\":{}}");
    try writeTmpFile(tts_dir, "tts_browser_onnx_meta.json", "{}");
    try writeTmpFile(tts_dir, "tokenizer.model", "spm");
    try writeTmpFile(codec_dir, "codec_browser_onnx_meta.json", "{\"codec_config\":{\"sample_rate\":48000,\"channels\":2}}");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var model = try normalizeModel(std.testing.allocator, root_path, .auto);
    defer model.deinit();

    try std.testing.expectEqual(types.ProviderKey.moss_tts_nano_tts, model.provider_key);
    try std.testing.expectEqual(types.Modality.tts, model.descriptor.modality);
    try std.testing.expectEqualStrings("moss_tts_nano", model.descriptor.family);
    try std.testing.expectEqual(types.RuntimeSupportStatus.degraded, model.support.status);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
