const backend_mod = @import("backend.zig");
const bert = @import("providers/bert.zig");
const chandra = @import("providers/chandra.zig");
const generic = @import("providers/generic.zig");
const moss_tts_nano = @import("providers/moss_tts_nano.zig");
const qwen3 = @import("providers/qwen3.zig");
const swiftocr = @import("providers/swiftocr.zig");
const types = @import("../types.zig");
const yolo = @import("providers/yolo.zig");
const std = @import("std");

pub const RuntimeBackend = backend_mod.RuntimeBackend;

const builtin_backends = [_]*const RuntimeBackend{
    &qwen3.backend,
    &bert.backend,
    &yolo.backend,
    &swiftocr.backend,
    &chandra.backend,
    &moss_tts_nano.backend,
    &generic.backend,
};

var dynamic_backends: std.ArrayListUnmanaged(*const RuntimeBackend) = .empty;

pub fn builtinBackends() []const *const RuntimeBackend {
    return &builtin_backends;
}

pub fn register(runtime_backend: *const RuntimeBackend) !void {
    if (findInSlice(dynamic_backends.items, runtime_backend.provider_key)) |index| {
        dynamic_backends.items[index] = runtime_backend;
        return;
    }
    try dynamic_backends.append(std.heap.page_allocator, runtime_backend);
}

pub fn unregister(provider_key: types.ProviderKey) bool {
    if (findInSlice(dynamic_backends.items, provider_key)) |index| {
        _ = dynamic_backends.swapRemove(index);
        return true;
    }
    return false;
}

pub fn registeredBackendsAlloc(allocator: std.mem.Allocator) ![]const *const RuntimeBackend {
    const combined = try allocator.alloc(*const RuntimeBackend, builtin_backends.len + dynamic_backends.items.len);
    @memcpy(combined[0..builtin_backends.len], &builtin_backends);
    @memcpy(combined[builtin_backends.len..], dynamic_backends.items);
    return combined;
}

pub fn findByKey(provider_key: types.ProviderKey) ?*const RuntimeBackend {
    if (findInSlice(dynamic_backends.items, provider_key)) |index| {
        return dynamic_backends.items[index];
    }
    for (builtin_backends) |runtime_backend| {
        if (runtime_backend.provider_key == provider_key) return runtime_backend;
    }
    return null;
}

fn findInSlice(backends: []const *const RuntimeBackend, provider_key: types.ProviderKey) ?usize {
    for (backends, 0..) |runtime_backend, index| {
        if (runtime_backend.provider_key == provider_key) return index;
    }
    return null;
}

test "runtime backend registry resolves builtin backends" {
    try @import("std").testing.expect(findByKey(.qwen3_text) != null);
    try @import("std").testing.expect(findByKey(.bert_text) != null);
    try @import("std").testing.expect(findByKey(.chandra_ocr) != null);
    try @import("std").testing.expect(findByKey(.moss_tts_nano_tts) != null);
    try @import("std").testing.expect(findByKey(.generic) != null);
}

test "runtime backend registry supports dynamic backend registration" {
    const custom = RuntimeBackend{
        .provider_key = .generic,
        .execute_fn = struct {
            fn run(
                _: std.mem.Allocator,
                _: *const @import("../model/handle.zig").ModelHandle,
                _: types.RuntimeRequest,
            ) !types.RuntimeResult {
                return .{};
            }
        }.run,
    };
    _ = unregister(.generic);
    try register(&custom);
    defer _ = unregister(.generic);
    try @import("std").testing.expect(findByKey(.generic) == &custom);
}
