const std = @import("std");
const handle_mod = @import("../model/handle.zig");
const normalized = @import("../model/resolver/normalized_model.zig");
const types = @import("../types.zig");

pub const OpenState = struct {
    provider_key: types.ProviderKey,
    model_dir: []u8,

    pub fn create(
        allocator: std.mem.Allocator,
        provider_key: types.ProviderKey,
        model: *const normalized.NormalizedModel,
    ) !*OpenState {
        const state = try allocator.create(OpenState);
        errdefer allocator.destroy(state);
        state.* = .{
            .provider_key = provider_key,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        };
        return state;
    }

    pub fn destroy(self: *OpenState, allocator: std.mem.Allocator) void {
        allocator.free(self.model_dir);
        allocator.destroy(self);
    }
};

pub fn openBasicState(
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) !?*anyopaque {
    return try OpenState.create(allocator, model.provider_key, model);
}

pub fn deinitBasicState(allocator: std.mem.Allocator, state: ?*anyopaque) void {
    const raw = state orelse return;
    const typed: *OpenState = @ptrCast(@alignCast(raw));
    typed.destroy(allocator);
}

pub const OpenFn = *const fn (
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) anyerror!?*anyopaque;

pub const DeinitFn = *const fn (
    allocator: std.mem.Allocator,
    state: ?*anyopaque,
) void;

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
    open_fn: ?OpenFn = null,
    deinit_fn: ?DeinitFn = null,
    execute_fn: ExecuteFn,
    execute_batch_fn: ?ExecuteBatchFn = null,

    pub fn open(
        self: *const RuntimeBackend,
        allocator: std.mem.Allocator,
        model: *const normalized.NormalizedModel,
    ) !?*anyopaque {
        if (self.open_fn) |open_fn| return try open_fn(allocator, model);
        return null;
    }

    pub fn deinitState(
        self: *const RuntimeBackend,
        allocator: std.mem.Allocator,
        state: ?*anyopaque,
    ) void {
        if (self.deinit_fn) |deinit_fn| deinit_fn(allocator, state);
    }

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
