const std = @import("std");
const kinetix = @import("../engine/kinetix.zig");

const backend = kinetix.artifacts.backend;
const registry_mod = kinetix.registry;
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
