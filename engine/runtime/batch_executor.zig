const std = @import("std");
const adapter_mod = @import("../adapter/adapter.zig");
const registry_mod = @import("../registry/registry.zig");
const scheduler_mod = @import("../scheduler/scheduler.zig");
const task = @import("../core/task.zig");

pub const RequestExecutionResult = struct {
    request_index: usize,
    result: adapter_mod.ExecutionResult,
};

pub const ExecutedBatch = struct {
    adapter_id: []const u8,
    execution: task.ExecutionMode,
    supports_batching: bool,
    execute_path: adapter_mod.BatchExecutionPath,
    request_results: []RequestExecutionResult,

    pub fn len(self: ExecutedBatch) usize {
        return self.request_results.len;
    }

    pub fn acceptedCount(self: ExecutedBatch) usize {
        var accepted: usize = 0;
        for (self.request_results) |result| {
            accepted += @intFromBool(result.result.submission.accepted);
        }
        return accepted;
    }

    pub fn commonOrigin(self: ExecutedBatch) ?adapter_mod.ExecutionOrigin {
        if (self.request_results.len == 0) return null;

        const origin = self.request_results[0].result.origin;
        for (self.request_results[1..]) |result| {
            if (result.result.origin != origin) return null;
        }
        return origin;
    }

    pub fn commonNote(self: ExecutedBatch) ?adapter_mod.ExecutionNote {
        if (self.request_results.len == 0) return null;

        const note = self.request_results[0].result.note;
        for (self.request_results[1..]) |result| {
            if (result.result.note != note) return null;
        }
        return note;
    }
};

pub const BatchExecutionReport = struct {
    allocator: std.mem.Allocator,
    batches: []ExecutedBatch,

    pub fn deinit(self: *BatchExecutionReport) void {
        for (self.batches) |batch| {
            for (batch.request_results) |*result| result.result.deinit(self.allocator);
            self.allocator.free(batch.request_results);
        }
        self.allocator.free(self.batches);
        self.* = undefined;
    }

    pub fn totalRequests(self: BatchExecutionReport) usize {
        var total: usize = 0;
        for (self.batches) |batch| total += batch.len();
        return total;
    }

    pub fn totalAccepted(self: BatchExecutionReport) usize {
        var total: usize = 0;
        for (self.batches) |batch| total += batch.acceptedCount();
        return total;
    }
};

pub fn execute(
    allocator: std.mem.Allocator,
    registry: *const registry_mod.Registry,
    requests: []const task.TaskRequest,
    batch_plan: *const scheduler_mod.BatchPlan,
) !BatchExecutionReport {
    const batches = try allocator.alloc(ExecutedBatch, batch_plan.batches.len);
    errdefer allocator.free(batches);

    for (batch_plan.batches, batches) |dispatch_batch, *executed_batch| {
        const entry = registry.findById(dispatch_batch.adapter_id) orelse return error.AdapterNotFound;
        const batch_requests = try allocator.alloc(task.TaskRequest, dispatch_batch.request_indices.len);
        defer allocator.free(batch_requests);
        const results = try allocator.alloc(RequestExecutionResult, dispatch_batch.request_indices.len);
        errdefer allocator.free(results);

        for (dispatch_batch.request_indices, batch_requests) |request_index, *request| {
            if (request_index >= requests.len) return error.InvalidRequestIndex;
            request.* = requests[request_index];
        }

        const results_batch = try entry.adapter.executeBatch(allocator, batch_requests);
        defer allocator.free(results_batch);
        if (results_batch.len != dispatch_batch.request_indices.len) return error.InvalidBatchSubmissionCount;

        for (dispatch_batch.request_indices, results_batch, results) |request_index, execution_result, *result| {
            result.* = .{
                .request_index = request_index,
                .result = execution_result,
            };
        }

        executed_batch.* = .{
            .adapter_id = dispatch_batch.adapter_id,
            .execution = dispatch_batch.execution,
            .supports_batching = dispatch_batch.supports_batching,
            .execute_path = entry.adapter.batchExecutePath(batch_requests.len),
            .request_results = results,
        };
    }

    return .{
        .allocator = allocator,
        .batches = batches,
    };
}
