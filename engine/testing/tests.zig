const std = @import("std");
const task = @import("../core/task.zig");
const adapter_mod = @import("../adapter/adapter.zig");
const registry_mod = @import("../registry/registry.zig");
const scheduler_mod = @import("../scheduler/scheduler.zig");

const MockState = struct {
    adapter_id: []const u8,
    submit_count: usize = 0,
};

fn submitMock(ctx: *anyopaque, spec: task.TaskSpec) !adapter_mod.Submission {
    const state: *MockState = @ptrCast(@alignCast(ctx));
    state.submit_count += 1;
    return .{
        .adapter_id = state.adapter_id,
        .accepted = true,
        .execution = spec.execution,
    };
}

const mock_vtable = adapter_mod.VTable{
    .submit = submitMock,
};

test "registry resolves task to matching modality adapter" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var text_state = MockState{ .adapter_id = "text.qwen3" };
    try registry.register(.{
        .ctx = &text_state,
        .descriptor = .{
            .id = "text.qwen3",
            .modality = .text,
            .supports_batching = true,
            .supports_streaming = true,
            .supported_operations = &.{ "generate", "embed" },
        },
        .vtable = &mock_vtable,
    });

    var vision_state = MockState{ .adapter_id = "vision.yolo" };
    try registry.register(.{
        .ctx = &vision_state,
        .descriptor = .{
            .id = "vision.yolo",
            .modality = .vision,
            .supports_batching = true,
            .supported_operations = &.{ "detect" },
        },
        .vtable = &mock_vtable,
    });

    const matched = registry.matchTask(.{
        .modality = .text,
        .operation = "generate",
        .model_family = "qwen3",
    }) orelse return error.ExpectedAdapterMatch;

    try std.testing.expectEqualStrings("text.qwen3", matched.adapter.descriptor.id);
}

test "scheduler plans and submits through explicit adapter id" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var state = MockState{ .adapter_id = "text.qwen3" };
    try registry.register(.{
        .ctx = &state,
        .descriptor = .{
            .id = "text.qwen3",
            .modality = .text,
            .supports_batching = true,
            .supports_streaming = true,
            .supported_operations = &.{ "generate", "embed" },
        },
        .vtable = &mock_vtable,
    });

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const spec = task.TaskSpec{
        .modality = .text,
        .operation = "generate",
        .model_family = "qwen3",
        .adapter_id = "text.qwen3",
        .execution = .stream,
    };

    const plan = try scheduler.plan(spec);
    try std.testing.expectEqualStrings("text.qwen3", plan.adapter_id);
    try std.testing.expectEqual(task.ExecutionMode.stream, plan.execution);
    try std.testing.expect(plan.supports_streaming);

    const submission = try scheduler.submit(spec);
    try std.testing.expect(submission.accepted);
    try std.testing.expectEqualStrings("text.qwen3", submission.adapter_id);
    try std.testing.expectEqual(@as(usize, 1), state.submit_count);
}

test "registry rejects duplicate adapter ids" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var state = MockState{ .adapter_id = "vision.yolo" };
    const adapter = adapter_mod.Adapter{
        .ctx = &state,
        .descriptor = .{
            .id = "vision.yolo",
            .modality = .vision,
            .supported_operations = &.{ "detect" },
        },
        .vtable = &mock_vtable,
    };

    try registry.register(adapter);
    try std.testing.expectError(error.AdapterAlreadyRegistered, registry.register(adapter));
}
