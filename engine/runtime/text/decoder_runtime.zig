const std = @import("std");
const backend_scheme = @import("backend_scheme.zig");
const legacy_runtime = @import("../../../legacy/zinfer/src/model/runtime/optimized_decoder.zig");

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
    return try Runtime.init(allocator, model_dir, scheme, thread_count);
}

test "bridge maps backend schemes without loss" {
    try std.testing.expectEqual(backend_scheme.Scheme.q8, schemeIdentity(.q8));
    try std.testing.expectEqual(backend_scheme.Scheme.bf16, schemeIdentity(.bf16));
}

fn schemeIdentity(scheme: backend_scheme.Scheme) backend_scheme.Scheme {
    return scheme;
}
