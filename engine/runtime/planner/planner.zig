const std = @import("std");
const handle_mod = @import("../model/handle.zig");
const types = @import("../types.zig");

pub const Planner = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Planner {
        return .{ .allocator = allocator };
    }

    pub fn plan(self: Planner, handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) !types.ExecutionPlan {
        try validateRequest(handle, request);

        const requests = try self.allocator.alloc(types.RuntimeRequest, 1);
        errdefer self.allocator.free(requests);
        requests[0] = request;

        const indices = try self.allocator.alloc(usize, 1);
        errdefer self.allocator.free(indices);
        indices[0] = 0;

        const batches = try self.allocator.alloc(types.PlanBatch, 1);
        errdefer self.allocator.free(batches);
        batches[0] = .{
            .allocator = self.allocator,
            .request_indices = indices,
            .operation = request.operation,
            .execution = request.execution,
            .allows_batching = handle.normalized.capabilities.supports_batch and request.allows_batching,
        };

        return .{
            .allocator = self.allocator,
            .model_id = handle.normalized.descriptor.id,
            .request_count = 1,
            .execution = request.execution,
            .path = choosePath(handle, request),
            .requests = requests,
            .batches = batches,
        };
    }

    pub fn planBatch(
        self: Planner,
        handle: *const handle_mod.ModelHandle,
        request: types.RuntimeBatchRequest,
    ) !types.ExecutionPlan {
        if (request.items.len == 0) return error.EmptyBatch;

        for (request.items) |item| {
            try validateRequest(handle, item);
        }

        const requests = try self.allocator.alloc(types.RuntimeRequest, request.items.len);
        errdefer self.allocator.free(requests);
        @memcpy(requests, request.items);
        const batches = try buildBatches(self.allocator, handle, request.items);

        return .{
            .allocator = self.allocator,
            .model_id = handle.normalized.descriptor.id,
            .request_count = request.items.len,
            .execution = request.items[0].execution,
            .path = choosePath(handle, request.items[0]),
            .requests = requests,
            .batches = batches,
        };
    }
};

fn validateRequest(handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) !void {
    if (!supportsOperation(handle, request.operation)) return error.OperationNotSupported;
    if (!acceptsInput(handle, types.inputKind(request.input))) return error.InvalidInputPayload;
    if (request.execution == .stream and !handle.normalized.capabilities.supports_stream) return error.StreamingNotSupported;
}

fn choosePath(handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) types.ExecutionPath {
    if (request.generation.native_execution and handle.normalized.capabilities.supports_native_exec) return .native;
    return .shared;
}

fn supportsOperation(handle: *const handle_mod.ModelHandle, operation: []const u8) bool {
    for (handle.normalized.capabilities.supported_operations) |supported| {
        if (std.mem.eql(u8, supported, operation)) return true;
    }
    return false;
}

fn acceptsInput(handle: *const handle_mod.ModelHandle, kind: types.InputKind) bool {
    for (handle.normalized.capabilities.accepted_inputs) |accepted| {
        if (accepted == kind) return true;
    }
    return kind == .none;
}

fn buildBatches(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    items: []const types.RuntimeRequest,
) ![]types.PlanBatch {
    const Builder = struct {
        operation: []const u8,
        execution: types.ExecutionMode,
        input_tag: std.meta.Tag(types.InputPayload),
        generation_max_tokens: ?usize,
        native_execution: bool,
        allows_batching: bool,
        indices: std.ArrayListUnmanaged(usize) = .empty,

        fn deinit(self: *@This(), alloc: std.mem.Allocator) void {
            self.indices.deinit(alloc);
        }
    };

    var builders = std.ArrayListUnmanaged(Builder).empty;
    defer {
        for (builders.items) |*builder| builder.deinit(allocator);
        builders.deinit(allocator);
    }

    for (items, 0..) |item, index| {
        const batching_allowed = handle.normalized.capabilities.supports_batch and item.allows_batching;
        const payload_tag = std.meta.activeTag(item.input);

        if (batching_allowed) {
            for (builders.items) |*builder| {
                if (!builder.allows_batching) continue;
                if (builder.execution != item.execution) continue;
                if (builder.input_tag != payload_tag) continue;
                if (builder.generation_max_tokens != item.generation.max_tokens) continue;
                if (builder.native_execution != item.generation.native_execution) continue;
                if (!std.mem.eql(u8, builder.operation, item.operation)) continue;
                try builder.indices.append(allocator, index);
                break;
            } else {
                var builder = Builder{
                    .operation = item.operation,
                    .execution = item.execution,
                    .input_tag = payload_tag,
                    .generation_max_tokens = item.generation.max_tokens,
                    .native_execution = item.generation.native_execution,
                    .allows_batching = true,
                };
                try builder.indices.append(allocator, index);
                try builders.append(allocator, builder);
            }
            continue;
        }

        var builder = Builder{
            .operation = item.operation,
            .execution = item.execution,
            .input_tag = payload_tag,
            .generation_max_tokens = item.generation.max_tokens,
            .native_execution = item.generation.native_execution,
            .allows_batching = false,
        };
        try builder.indices.append(allocator, index);
        try builders.append(allocator, builder);
    }

    const batches = try allocator.alloc(types.PlanBatch, builders.items.len);
    errdefer allocator.free(batches);

    var initialized: usize = 0;
    errdefer {
        for (batches[0..initialized]) |*batch| batch.deinit();
    }

    for (builders.items, batches) |*builder, *batch| {
        batch.* = .{
            .allocator = allocator,
            .request_indices = try builder.indices.toOwnedSlice(allocator),
            .operation = builder.operation,
            .execution = builder.execution,
            .allows_batching = builder.allows_batching,
        };
        initialized += 1;
    }

    return batches;
}
