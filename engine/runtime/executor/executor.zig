const std = @import("std");
const adapter_mod = @import("../../adapter/adapter.zig");
const task = @import("../../core/task.zig");
const handle_mod = @import("../model/handle.zig");
const text_runtime = @import("../text/text.zig");
const types = @import("../types.zig");

pub const Executor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Executor {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeResult {
        if (plan.request_count != 1 or plan.requests.len != 1) return error.InvalidExecutionPlan;

        return switch (handle.normalized.provider_key) {
            .qwen3_text => try executeQwen3(self.allocator, handle, plan.requests[0]),
            else => error.RuntimeExecutionNotImplemented,
        };
    }
};

fn executeQwen3(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    var legacy_result = try text_runtime.native_dispatch.executeSingle(
        allocator,
        text_runtime.native_dispatch.NativeBatchBridge,
        handle.normalized.artifacts.model_dir,
        .auto,
        .{
            .adapter_id = handle.normalized.descriptor.id,
            .accepted = true,
            .execution = request.execution,
        },
        buildTaskRequest(handle, request),
    );
    errdefer legacy_result.deinit(allocator);

    const output = legacy_result.output;
    legacy_result.output = .none;
    return .{
        .origin = legacy_result.origin,
        .note = @tagName(legacy_result.note),
        .output = output,
    };
}

fn buildTaskRequest(handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) task.TaskRequest {
    return .{
        .spec = .{
            .modality = handle.normalized.descriptor.modality,
            .operation = request.operation,
            .model_family = handle.normalized.descriptor.family,
            .adapter_id = handle.normalized.descriptor.id,
            .execution = request.execution,
        },
        .input = request.input,
        .generation = request.generation,
    };
}
