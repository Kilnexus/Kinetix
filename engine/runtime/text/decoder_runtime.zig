const std = @import("std");
const backend_scheme = @import("backend_scheme.zig");
const legacy_runtime = @import("../../../legacy/zinfer/src/model/runtime/optimized_decoder.zig");
const legacy_backend = @import("../../../legacy/zinfer/src/tensor/backends/backend.zig");

pub const BatchRuntime = legacy_runtime.BatchRuntime;
pub const BatchDecodeStats = legacy_runtime.BatchDecodeStats;
pub const Runtime = legacy_runtime.Runtime;
pub const Workspace = legacy_runtime.Workspace;

pub fn initRuntime(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    scheme: backend_scheme.Scheme,
    thread_count: ?usize,
) !Runtime {
    return try Runtime.init(allocator, model_dir, mapBackendScheme(scheme), thread_count);
}

fn mapBackendScheme(scheme: backend_scheme.Scheme) legacy_backend.Scheme {
    return switch (scheme) {
        .auto => .auto,
        .bf16 => .bf16,
        .q8 => .q8,
        .q6 => .q6,
        .q4 => .q4,
    };
}

test "bridge maps backend schemes without loss" {
    try std.testing.expectEqual(legacy_backend.Scheme.q8, mapBackendScheme(.q8));
    try std.testing.expectEqual(legacy_backend.Scheme.bf16, mapBackendScheme(.bf16));
}
