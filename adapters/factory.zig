const std = @import("std");
const kinetix = @import("engine_root");

const backend = kinetix.artifacts.backend;
const registry_mod = kinetix.registry;
const runtime_model = kinetix.runtime.model;
const runtime_types = kinetix.runtime.types;
const ocr_mod = @import("ocr/ocr.zig");
const text_mod = @import("text/text.zig");
const vision_mod = @import("vision/vision.zig");

pub const ManagedAdapter = union(enum) {
    text: text_mod.TextAdapter,
    vision: vision_mod.VisionAdapter,
    ocr: ocr_mod.OCRAdapter,

    pub fn deinit(self: *ManagedAdapter) void {
        switch (self.*) {
            .text => |*adapter| adapter.deinit(),
            .vision => |*adapter| adapter.deinit(),
            .ocr => |*adapter| adapter.deinit(),
        }
    }

    pub fn registerInto(self: *ManagedAdapter, registry: *registry_mod.Registry) !void {
        switch (self.*) {
            .text => |*adapter| try adapter.registerInto(registry),
            .vision => |*adapter| try adapter.registerInto(registry),
            .ocr => |*adapter| try adapter.registerInto(registry),
        }
    }

    pub fn descriptorId(self: *const ManagedAdapter) []const u8 {
        return switch (self.*) {
            .text => |adapter| adapter.descriptor.id,
            .vision => |adapter| adapter.descriptor.id,
            .ocr => |adapter| adapter.descriptor.id,
        };
    }

    pub fn descriptor(self: *const ManagedAdapter) kinetix.adapter.Descriptor {
        return switch (self.*) {
            .text => |adapter| adapter.descriptor,
            .vision => |adapter| adapter.descriptor,
            .ocr => |adapter| adapter.descriptor,
        };
    }
};

pub fn initAuto(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
) !ManagedAdapter {
    var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
    defer catalog.deinit();

    if (catalog.has(.graph_json) and catalog.has(.weights_bin)) {
        return .{ .vision = try vision_mod.VisionAdapter.init(allocator, model_dir) };
    }
    if (catalog.has(.config) and catalog.has(.tokenizer_json) and (catalog.resolveAutoScheme() != .auto or catalog.has(.safetensors))) {
        return .{ .text = try text_mod.TextAdapter.init(allocator, model_dir, preferred_weights) };
    }
    if (catalog.has(.ocr_model)) {
        return .{ .ocr = try ocr_mod.OCRAdapter.init(allocator, model_dir) };
    }

    return error.UnsupportedModelDirectory;
}

pub fn initForModelHandle(
    allocator: std.mem.Allocator,
    handle: *const runtime_model.ModelHandle,
    preferred_weights: backend.WeightScheme,
) !ManagedAdapter {
    const model_dir = handle.normalized.artifacts.model_dir;
    return switch (handle.normalized.provider_key) {
        .qwen3_text, .bert_text => .{
            .text = try text_mod.TextAdapter.init(allocator, model_dir, preferred_weights),
        },
        .yolo_vision => .{
            .vision = try vision_mod.VisionAdapter.init(allocator, model_dir),
        },
        .swiftocr_ocr => .{
            .ocr = try ocr_mod.OCRAdapter.init(allocator, model_dir),
        },
        .generic => switch (handle.normalized.descriptor.modality) {
            .text, .multimodal => .{
                .text = try text_mod.TextAdapter.init(allocator, model_dir, preferred_weights),
            },
            .vision => .{
                .vision = try vision_mod.VisionAdapter.init(allocator, model_dir),
            },
            .ocr => .{
                .ocr = try ocr_mod.OCRAdapter.init(allocator, model_dir),
            },
            else => error.UnsupportedNormalizedProvider,
        },
    };
}

test "factory can bridge from unified runtime model handle to legacy adapters" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var handle = runtime_model.ModelHandle{
        .allocator = std.testing.allocator,
        .normalized = .{
            .descriptor = .{
                .allocator = std.testing.allocator,
                .id = try std.testing.allocator.dupe(u8, "runtime.text.qwen3.demo"),
                .modality = .text,
                .family = try std.testing.allocator.dupe(u8, "qwen3"),
                .source_format = .huggingface_directory,
                .normalized_format = .text_decoder,
            },
            .artifacts = .{
                .allocator = std.testing.allocator,
                .model_dir = try std.testing.allocator.dupe(u8, root_path),
                .config_path = null,
                .tokenizer_path = null,
                .graph_path = null,
                .weights_path = null,
                .binary_weights_path = null,
                .ocr_model_path = null,
            },
            .capabilities = .{
                .supports_sync = true,
                .supports_batch = true,
                .supports_stream = true,
                .supports_native_exec = true,
            },
            .compat = try kinetix.runtime.compat.CompatibilityReport.supported(std.testing.allocator),
            .provider_key = .qwen3_text,
        },
    };
    defer handle.deinit();

    var managed = try initForModelHandle(std.testing.allocator, &handle, .auto);
    defer managed.deinit();

    try std.testing.expect(std.mem.startsWith(u8, managed.descriptorId(), "text.qwen3."));
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
