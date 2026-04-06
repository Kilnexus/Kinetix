const task = @import("../core/task.zig");
const adapter_mod = @import("../adapter/adapter.zig");
const registry_mod = @import("../registry/registry.zig");

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

    pub fn submit(self: *const Scheduler, spec: task.TaskSpec) !adapter_mod.Submission {
        const entry = self.registry.matchTask(spec) orelse return error.NoMatchingAdapter;
        _ = try self.plan(spec);
        return try entry.adapter.submit(spec);
    }
};
