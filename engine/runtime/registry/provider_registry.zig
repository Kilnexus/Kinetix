const std = @import("std");
const backend_registry = @import("../backend/registry.zig");
const types = @import("../types.zig");

pub const ProviderDescriptor = struct {
    key: types.ProviderKey,
    modality: types.Modality,
    family: []const u8,
};

const builtin_descriptors = blk: {
    const runtime_backends = backend_registry.builtinBackends();
    var projected: [runtime_backends.len]ProviderDescriptor = undefined;
    for (runtime_backends, 0..) |runtime_backend, index| {
        projected[index] = describeProvider(runtime_backend.provider_key);
    }
    break :blk projected;
};

pub fn descriptors() []const ProviderDescriptor {
    return &builtin_descriptors;
}

pub fn findByKey(key: types.ProviderKey) ?ProviderDescriptor {
    if (backend_registry.findByKey(key) == null) return null;
    return describeProvider(key);
}

pub fn findByFamily(modality: types.Modality, family: []const u8) ?ProviderDescriptor {
    for (descriptors()) |descriptor| {
        if (descriptor.modality != modality and descriptor.key != .generic) continue;
        if (std.mem.eql(u8, descriptor.family, family)) return descriptor;
    }
    return null;
}

fn describeProvider(key: types.ProviderKey) ProviderDescriptor {
    return switch (key) {
        .qwen3_text => .{ .key = .qwen3_text, .modality = .text, .family = "qwen3" },
        .bert_text => .{ .key = .bert_text, .modality = .text, .family = "bert" },
        .yolo_vision => .{ .key = .yolo_vision, .modality = .vision, .family = "yolo" },
        .swiftocr_ocr => .{ .key = .swiftocr_ocr, .modality = .ocr, .family = "swiftocr" },
        .chandra_ocr => .{ .key = .chandra_ocr, .modality = .ocr, .family = "chandra" },
        .generic => .{ .key = .generic, .modality = .multimodal, .family = "generic" },
    };
}

test "provider registry resolves builtin providers by key" {
    const descriptor = findByKey(.qwen3_text) orelse return error.ExpectedProvider;
    try std.testing.expectEqual(types.Modality.text, descriptor.modality);
    try std.testing.expectEqualStrings("qwen3", descriptor.family);
}

test "provider registry projects the unified backend registry" {
    try std.testing.expectEqual(backend_registry.builtinBackends().len, descriptors().len);
    try std.testing.expect(findByKey(.bert_text) != null);
    try std.testing.expect(findByKey(.generic) != null);
}
