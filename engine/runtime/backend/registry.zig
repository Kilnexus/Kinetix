const backend_mod = @import("backend.zig");
const chandra = @import("providers/chandra.zig");
const qwen3 = @import("providers/qwen3.zig");
const swiftocr = @import("providers/swiftocr.zig");
const types = @import("../types.zig");
const yolo = @import("providers/yolo.zig");

pub const RuntimeBackend = backend_mod.RuntimeBackend;

const builtin_backends = [_]*const RuntimeBackend{
    &qwen3.backend,
    &yolo.backend,
    &swiftocr.backend,
    &chandra.backend,
};

pub fn builtinBackends() []const *const RuntimeBackend {
    return &builtin_backends;
}

pub fn findByKey(provider_key: types.ProviderKey) ?*const RuntimeBackend {
    for (builtin_backends) |runtime_backend| {
        if (runtime_backend.provider_key == provider_key) return runtime_backend;
    }
    return null;
}

test "runtime backend registry resolves builtin backends" {
    try @import("std").testing.expect(findByKey(.qwen3_text) != null);
    try @import("std").testing.expect(findByKey(.chandra_ocr) != null);
}
