const std = @import("std");
const task = @import("../core/task.zig");
const adapter_mod = @import("../adapter/adapter.zig");
const registry_mod = @import("../registry/registry.zig");

pub const RuntimePool = @import("runtime_pool.zig").RuntimePool;

pub const QueueKey = struct {
    modality: task.Modality,
    operation: []const u8,
    model_family: []const u8,
};

pub const DispatchPlan = struct {
    adapter_id: []const u8,
    queue: QueueKey,
    execution: task.ExecutionMode,
    supports_batching: bool,
    supports_streaming: bool,
};

pub const DispatchBatch = struct {
    adapter_id: []const u8,
    queue: QueueKey,
    execution: task.ExecutionMode,
    supports_batching: bool,
    supports_streaming: bool,
    input_tag: std.meta.Tag(task.InputPayload),
    request_indices: []usize,

    pub fn len(self: DispatchBatch) usize {
        return self.request_indices.len;
    }
};

pub const BatchPlan = struct {
    allocator: std.mem.Allocator,
    batches: []DispatchBatch,

    pub fn deinit(self: *BatchPlan) void {
        for (self.batches) |batch| {
            self.allocator.free(batch.request_indices);
        }
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

pub const Scheduler = struct {
    registry: *const registry_mod.Registry,

    pub fn init(registry: *const registry_mod.Registry) Scheduler {
        return .{
            .registry = registry,
        };
    }

    pub fn plan(self: *const Scheduler, spec: task.TaskSpec) !DispatchPlan {
        const entry = self.registry.matchTask(spec) orelse return error.NoMatchingAdapter;
        if (spec.execution == .stream and !entry.adapter.descriptor.supports_streaming) {
            return error.StreamingNotSupported;
        }

        return .{
            .adapter_id = entry.adapter.descriptor.id,
            .queue = .{
                .modality = spec.modality,
                .operation = spec.operation,
                .model_family = spec.model_family,
            },
            .execution = spec.execution,
            .supports_batching = entry.adapter.descriptor.supports_batching,
            .supports_streaming = entry.adapter.descriptor.supports_streaming,
        };
    }

    pub fn planRequest(self: *const Scheduler, request: task.TaskRequest) !DispatchPlan {
        return try self.plan(request.spec);
    }

    pub fn submit(self: *const Scheduler, request: task.TaskRequest) !adapter_mod.Submission {
        const entry = self.registry.matchTask(request.spec) orelse return error.NoMatchingAdapter;
        _ = try self.planRequest(request);
        return try entry.adapter.submit(request);
    }

    pub fn submitSpec(self: *const Scheduler, spec: task.TaskSpec) !adapter_mod.Submission {
        return try self.submit(.{ .spec = spec });
    }

    pub fn planBatches(self: *const Scheduler, allocator: std.mem.Allocator, requests: []const task.TaskRequest) !BatchPlan {
        const Builder = struct {
            adapter_id: []const u8,
            queue: QueueKey,
            execution: task.ExecutionMode,
            supports_batching: bool,
            supports_streaming: bool,
            input_tag: std.meta.Tag(task.InputPayload),
            indices: std.ArrayListUnmanaged(usize) = .empty,

            fn deinit(builder: *@This(), alloc: std.mem.Allocator) void {
                builder.indices.deinit(alloc);
            }
        };

        var builders = std.ArrayListUnmanaged(Builder).empty;
        defer {
            for (builders.items) |*builder| builder.deinit(allocator);
            builders.deinit(allocator);
        }

        for (requests, 0..) |request, index| {
            const dispatch_plan = try self.planRequest(request);
            const payload_tag = std.meta.activeTag(request.input);
            const batching_allowed = dispatch_plan.supports_batching and request.spec.allows_batching;

            if (batching_allowed) {
                for (builders.items) |*builder| {
                    if (!builder.supports_batching) continue;
                    if (builder.execution != dispatch_plan.execution) continue;
                    if (builder.input_tag != payload_tag) continue;
                    if (!std.mem.eql(u8, builder.adapter_id, dispatch_plan.adapter_id)) continue;
                    if (builder.queue.modality != dispatch_plan.queue.modality) continue;
                    if (!std.mem.eql(u8, builder.queue.operation, dispatch_plan.queue.operation)) continue;
                    if (!std.mem.eql(u8, builder.queue.model_family, dispatch_plan.queue.model_family)) continue;
                    try builder.indices.append(allocator, index);
                    break;
                } else {
                    var builder = Builder{
                        .adapter_id = dispatch_plan.adapter_id,
                        .queue = dispatch_plan.queue,
                        .execution = dispatch_plan.execution,
                        .supports_batching = true,
                        .supports_streaming = dispatch_plan.supports_streaming,
                        .input_tag = payload_tag,
                    };
                    try builder.indices.append(allocator, index);
                    try builders.append(allocator, builder);
                }
                continue;
            }

            var builder = Builder{
                .adapter_id = dispatch_plan.adapter_id,
                .queue = dispatch_plan.queue,
                .execution = dispatch_plan.execution,
                .supports_batching = false,
                .supports_streaming = dispatch_plan.supports_streaming,
                .input_tag = payload_tag,
            };
            try builder.indices.append(allocator, index);
            try builders.append(allocator, builder);
        }

        const batches = try allocator.alloc(DispatchBatch, builders.items.len);
        errdefer allocator.free(batches);

        for (builders.items, batches) |*builder, *batch| {
            batch.* = .{
                .adapter_id = builder.adapter_id,
                .queue = builder.queue,
                .execution = builder.execution,
                .supports_batching = builder.supports_batching,
                .supports_streaming = builder.supports_streaming,
                .input_tag = builder.input_tag,
                .request_indices = try builder.indices.toOwnedSlice(allocator),
            };
        }

        return .{
            .allocator = allocator,
            .batches = batches,
        };
    }
};
