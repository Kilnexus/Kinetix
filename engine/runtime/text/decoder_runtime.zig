const std = @import("std");
const backend_scheme = @import("backend_scheme.zig");
const optimized_batch = @import("optimized_decoder/batch.zig");
const optimized_runtime = @import("optimized_decoder/runtime.zig");
const optimized_workspace = @import("optimized_decoder/workspace.zig");

pub const BatchRuntime = optimized_batch.BatchRuntime;
pub const BatchDecodeStats = optimized_batch.DecodeStats;
pub const Runtime = optimized_runtime.Runtime;
pub const Workspace = optimized_workspace.Workspace;

pub fn initRuntime(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    scheme: backend_scheme.Scheme,
    thread_count: ?usize,
) !Runtime {
    return try Runtime.init(allocator, model_dir, scheme, thread_count);
}

test "runtime maps backend schemes without loss" {
    try std.testing.expectEqual(backend_scheme.Scheme.q8, schemeIdentity(.q8));
    try std.testing.expectEqual(backend_scheme.Scheme.bf16, schemeIdentity(.bf16));
}

fn schemeIdentity(scheme: backend_scheme.Scheme) backend_scheme.Scheme {
    return scheme;
}
