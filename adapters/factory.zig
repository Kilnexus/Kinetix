const std = @import("std");
const kinetix = @import("engine_root");

const backend = kinetix.artifacts.backend;
const registry_mod = kinetix.registry;
const runtime_compat = kinetix.runtime.compat;
const runtime_model = kinetix.runtime.model;
const runtime_types = kinetix.runtime.types;
const runtime_session = kinetix.runtime.session;
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
    var normalized = try runtime_compat.normalizeModel(allocator, model_dir, preferred_weights);
    defer normalized.deinit();
    return try initForNormalizedModel(allocator, &normalized, preferred_weights);
}

pub fn initForModelHandle(
    allocator: std.mem.Allocator,
    handle: *const runtime_model.ModelHandle,
    preferred_weights: backend.WeightScheme,
) !ManagedAdapter {
    return try initForNormalizedModel(allocator, &handle.normalized, preferred_weights);
}

pub fn initForNormalizedModel(
    allocator: std.mem.Allocator,
    normalized: *const runtime_compat.NormalizedModel,
    preferred_weights: backend.WeightScheme,
) !ManagedAdapter {
    const model_dir = normalized.artifacts.model_dir;
    return switch (normalized.provider_key) {
        .qwen3_text, .bert_text => try initForModality(allocator, .text, model_dir, preferred_weights),
        .yolo_vision => try initForModality(allocator, .vision, model_dir, preferred_weights),
        .swiftocr_ocr => try initForModality(allocator, .ocr, model_dir, preferred_weights),
        .generic => try initForModality(allocator, normalized.descriptor.modality, model_dir, preferred_weights),
    };
}

fn initForModality(
    allocator: std.mem.Allocator,
    modality: runtime_types.Modality,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
) !ManagedAdapter {
    return switch (modality) {
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

    var session = runtime_session.RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
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
