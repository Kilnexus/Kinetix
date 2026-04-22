const std = @import("std");
const handle_mod = @import("../model/handle.zig");
const types = @import("../types.zig");

pub const ExecuteFn = *const fn (
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) anyerror!types.RuntimeResult;

pub const ExecuteBatchFn = *const fn (
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    requests: []const types.RuntimeRequest,
) anyerror!types.RuntimeBatchResults;

pub const RuntimeBackend = struct {
    provider_key: types.ProviderKey,
    execute_fn: ExecuteFn,
    execute_batch_fn: ?ExecuteBatchFn = null,

    pub fn execute(
        self: *const RuntimeBackend,
        allocator: std.mem.Allocator,
        handle: *const handle_mod.ModelHandle,
        request: types.RuntimeRequest,
    ) !types.RuntimeResult {
        return try self.execute_fn(allocator, handle, request);
    }

    pub fn executeBatch(
        self: *const RuntimeBackend,
        allocator: std.mem.Allocator,
        handle: *const handle_mod.ModelHandle,
        requests: []const types.RuntimeRequest,
    ) !types.RuntimeBatchResults {
        if (self.execute_batch_fn) |execute_batch| {
            return try execute_batch(allocator, handle, requests);
        }
        return try self.executeBatchSequential(allocator, handle, requests);
    }

    fn executeBatchSequential(
        self: *const RuntimeBackend,
        allocator: std.mem.Allocator,
        handle: *const handle_mod.ModelHandle,
        requests: []const types.RuntimeRequest,
    ) !types.RuntimeBatchResults {
        const results = try allocator.alloc(types.RuntimeResult, requests.len);
        errdefer allocator.free(results);

        var initialized: usize = 0;
        errdefer {
            for (results[0..initialized]) |*result| result.deinit(allocator);
            allocator.free(results);
        }

        for (requests, results) |request, *result| {
            result.* = try self.execute(allocator, handle, request);
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .items = results,
        };
    }
};
