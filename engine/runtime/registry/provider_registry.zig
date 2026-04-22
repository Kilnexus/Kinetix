const std = @import("std");
const backend_registry = @import("../backend/registry.zig");
const family_registry = @import("../families/registry.zig");
const types = @import("../types.zig");

pub const ProviderDescriptor = family_registry.ProviderDescriptor;

const builtin_descriptors = blk: {
    const families = family_registry.builtinFamilies();
    var projected: [families.len]ProviderDescriptor = undefined;
    for (families, 0..) |family, index| {
        projected[index] = family.descriptor;
    }
    break :blk projected;
};

pub fn descriptors() []const ProviderDescriptor {
    return &builtin_descriptors;
}

pub fn findByKey(key: types.ProviderKey) ?ProviderDescriptor {
    if (backend_registry.findByKey(key) == null) return null;
    const family = family_registry.findBuiltinByKey(key) orelse return null;
    return family.descriptor;
}

pub fn findByFamily(modality: types.Modality, family: []const u8) ?ProviderDescriptor {
    for (descriptors()) |descriptor| {
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

test "provider registry projects the unified backend registry" {
    try std.testing.expectEqual(backend_registry.builtinBackends().len, descriptors().len);
    try std.testing.expect(findByKey(.bert_text) != null);
    try std.testing.expect(findByKey(.generic) != null);
}
