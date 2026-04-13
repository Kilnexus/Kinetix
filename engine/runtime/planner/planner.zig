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
            .allows_batching = handle.normalized.capabilities.supports_batch,
        };

        return .{
            .allocator = self.allocator,
            .model_id = handle.normalized.descriptor.id,
            .request_count = 1,
            .execution = request.execution,
            .path = choosePath(handle, request),
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

        const same_operation = allItemsShareOperation(request.items);
        const use_batching = handle.normalized.capabilities.supports_batch and same_operation;
        const batch_count: usize = if (use_batching) 1 else request.items.len;
        const batches = try self.allocator.alloc(types.PlanBatch, batch_count);
        errdefer self.allocator.free(batches);

        if (use_batching) {
            const indices = try self.allocator.alloc(usize, request.items.len);
            errdefer self.allocator.free(indices);
            for (indices, 0..) |*slot, index| slot.* = index;
            batches[0] = .{
                .allocator = self.allocator,
                .request_indices = indices,
                .operation = request.items[0].operation,
                .execution = request.items[0].execution,
                .allows_batching = true,
            };
        } else {
            var initialized: usize = 0;
            errdefer {
                for (batches[0..initialized]) |*batch| batch.deinit();
            }
            for (request.items, 0..) |item, index| {
                const indices = try self.allocator.alloc(usize, 1);
                indices[0] = index;
                batches[index] = .{
                    .allocator = self.allocator,
                    .request_indices = indices,
                    .operation = item.operation,
                    .execution = item.execution,
                    .allows_batching = false,
                };
                initialized += 1;
            }
        }

        return .{
            .allocator = self.allocator,
            .model_id = handle.normalized.descriptor.id,
            .request_count = request.items.len,
            .execution = request.items[0].execution,
            .path = choosePath(handle, request.items[0]),
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

fn allItemsShareOperation(items: []const types.RuntimeRequest) bool {
    for (items[1..]) |item| {
        if (!std.mem.eql(u8, items[0].operation, item.operation)) return false;
        if (items[0].execution != item.execution) return false;
        if (items[0].generation.native_execution != item.generation.native_execution) return false;
        if (items[0].generation.max_tokens != item.generation.max_tokens) return false;
    }
    return true;
}
