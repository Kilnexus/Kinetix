const std = @import("std");
const backend_mod = @import("../backend.zig");
const task = @import("../../../core/task.zig");
const handle_mod = @import("../../model/handle.zig");
const text_shared = @import("../../providers/text_shared.zig");
const types = @import("../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .qwen3_text,
    .execute_fn = execute,
    .execute_batch_fn = executeBatch,
};

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    return try text_shared.executeQwenSingle(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        buildTaskRequest(handle, request),
    );
}

fn executeBatch(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    requests: []const types.RuntimeRequest,
) !types.RuntimeBatchResults {
    const task_requests = try allocator.alloc(task.TaskRequest, requests.len);
    defer allocator.free(task_requests);

    for (requests, task_requests) |request, *slot| {
        slot.* = buildTaskRequest(handle, request);
    }

    return try text_shared.executeQwenBatch(
        allocator,
        handle.normalized.artifacts.model_dir,
        .auto,
        handle.normalized.descriptor.id,
        task_requests,
    );
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
