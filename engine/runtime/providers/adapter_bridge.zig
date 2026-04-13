const std = @import("std");
const adapter_mod = @import("../../adapter/adapter.zig");
const task = @import("../../core/task.zig");
const handle_mod = @import("../model/handle.zig");
const session_mod = @import("../session/session.zig");
const text_shared = @import("text_shared.zig");
const types = @import("../types.zig");

pub fn executeSingle(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    adapter_id: []const u8,
    request: task.TaskRequest,
) !adapter_mod.ExecutionResult {
    var session = session_mod.RuntimeSession.init(allocator);
    defer session.deinit();

    var plan = try session.plan(handle, toRuntimeRequest(request));
    defer plan.deinit();

    var runtime_result = try session.execute(handle, &plan);
    defer runtime_result.deinit(allocator);

    const output = runtime_result.output;
    runtime_result.output = .none;
    return .{
        .submission = text_shared.buildSubmission(adapter_id, request.spec.execution),
        .origin = runtime_result.origin,
        .note = parseExecutionNote(runtime_result.note),
        .output = output,
    };
}

pub fn executeBatch(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    adapter_id: []const u8,
    requests: []const task.TaskRequest,
) ![]adapter_mod.ExecutionResult {
    const runtime_requests = try allocator.alloc(types.RuntimeRequest, requests.len);
    defer allocator.free(runtime_requests);

    for (requests, runtime_requests) |request, *slot| {
        slot.* = toRuntimeRequest(request);
    }

    var session = session_mod.RuntimeSession.init(allocator);
    defer session.deinit();

    var plan = try session.planBatch(handle, .{ .items = runtime_requests });
    defer plan.deinit();

    var runtime_results = try session.executeBatch(handle, &plan);
    defer runtime_results.deinit();

    const results = try allocator.alloc(adapter_mod.ExecutionResult, runtime_results.items.len);
    errdefer allocator.free(results);

    for (runtime_results.items, results, 0..) |*runtime_result, *result, index| {
        const output = runtime_result.output;
        runtime_result.output = .none;
        result.* = .{
            .submission = text_shared.buildSubmission(adapter_id, requests[index].spec.execution),
            .origin = runtime_result.origin,
            .note = parseExecutionNote(runtime_result.note),
            .output = output,
        };
    }

    return results;
}

pub fn toRuntimeRequest(request: task.TaskRequest) types.RuntimeRequest {
    return .{
        .operation = request.spec.operation,
        .input = request.input,
        .execution = request.spec.execution,
        .generation = request.generation,
    };
}

fn parseExecutionNote(value: []const u8) adapter_mod.ExecutionNote {
    if (std.mem.eql(u8, value, "validated_only")) return .validated_only;
    if (std.mem.eql(u8, value, "text_request_ready")) return .text_request_ready;
    if (std.mem.eql(u8, value, "text_native_qwen_single")) return .text_native_qwen_single;
    if (std.mem.eql(u8, value, "text_native_qwen_batch")) return .text_native_qwen_batch;
    if (std.mem.eql(u8, value, "vision_graph_ready")) return .vision_graph_ready;
    if (std.mem.eql(u8, value, "vision_shared_detect")) return .vision_shared_detect;
    if (std.mem.eql(u8, value, "ocr_model_ready")) return .ocr_model_ready;
    if (std.mem.eql(u8, value, "ocr_shared_infer")) return .ocr_shared_infer;
    return .none;
}
