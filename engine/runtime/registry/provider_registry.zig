const std = @import("std");
const types = @import("../types.zig");

pub const ProviderDescriptor = struct {
    key: types.ProviderKey,
    modality: types.Modality,
    family: []const u8,
};

const builtin_descriptors = [_]ProviderDescriptor{
    .{ .key = .qwen3_text, .modality = .text, .family = "qwen3" },
    .{ .key = .bert_text, .modality = .text, .family = "bert" },
    .{ .key = .yolo_vision, .modality = .vision, .family = "yolo" },
    .{ .key = .swiftocr_ocr, .modality = .ocr, .family = "swiftocr" },
    .{ .key = .generic, .modality = .multimodal, .family = "generic" },
};

pub fn descriptors() []const ProviderDescriptor {
    return &builtin_descriptors;
}

pub fn findByKey(key: types.ProviderKey) ?ProviderDescriptor {
    for (builtin_descriptors) |descriptor| {
        if (descriptor.key == key) return descriptor;
    }
    return null;
}

pub fn findByFamily(modality: types.Modality, family: []const u8) ?ProviderDescriptor {
    for (builtin_descriptors) |descriptor| {
        if (descriptor.modality != modality and descriptor.key != .generic) continue;
        if (std.mem.eql(u8, descriptor.family, family)) return descriptor;
    }
    return null;
}

test "provider registry resolves builtin providers by key" {
    const descriptor = findByKey(.qwen3_text) orelse return error.ExpectedProvider;
    try std.testing.expectEqual(types.Modality.text, descriptor.modality);
    try std.testing.expectEqualStrings("qwen3", descriptor.family);
}
