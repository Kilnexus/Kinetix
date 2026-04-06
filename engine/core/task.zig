const std = @import("std");

pub const Modality = enum {
    vision,
    ocr,
    text,
    video,
    tts,
    audio,
    multimodal,
};

pub const ExecutionMode = enum {
    sync,
    async,
    stream,
};

pub const TaskSpec = struct {
    modality: Modality,
    operation: []const u8,
    model_family: []const u8,
    adapter_id: ?[]const u8 = null,
    model_name: ?[]const u8 = null,
    execution: ExecutionMode = .sync,
    allows_batching: bool = true,
    priority: u8 = 0,
};

test "task spec defaults are stable" {
    const spec = TaskSpec{
        .modality = .text,
        .operation = "generate",
        .model_family = "qwen3",
    };

    try std.testing.expectEqual(ExecutionMode.sync, spec.execution);
    try std.testing.expect(spec.allows_batching);
    try std.testing.expectEqual(@as(u8, 0), spec.priority);
}
