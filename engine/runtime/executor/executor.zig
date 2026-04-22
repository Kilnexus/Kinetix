const std = @import("std");
const backend_registry = @import("../backend/registry.zig");
const handle_mod = @import("../model/handle.zig");
const types = @import("../types.zig");

pub const Executor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Executor {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeResult {
        if (plan.request_count != 1 or plan.requests.len != 1) return error.InvalidExecutionPlan;

        const runtime_backend = backend_registry.findByKey(handle.normalized.provider_key) orelse return error.RuntimeExecutionNotImplemented;
        return try runtime_backend.execute(self.allocator, handle, plan.requests[0]);
    }

    pub fn executeBatch(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeBatchResults {
        if (plan.request_count == 0 or plan.requests.len == 0) return error.InvalidExecutionPlan;

        const runtime_backend = backend_registry.findByKey(handle.normalized.provider_key) orelse return error.RuntimeExecutionNotImplemented;
        return try runtime_backend.executeBatch(self.allocator, handle, plan.requests);
    }
};

test "executor delegates provider selection to backend registry" {
    try std.testing.expect(backend_registry.findByKey(.qwen3_text) != null);
    try std.testing.expect(backend_registry.findByKey(.yolo_vision) != null);
    try std.testing.expect(backend_registry.findByKey(.swiftocr_ocr) != null);
    try std.testing.expect(backend_registry.findByKey(.chandra_ocr) != null);
}
