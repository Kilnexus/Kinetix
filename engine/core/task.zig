const std = @import("std");
const abi = @import("runtime_abi");

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
    operation_id: abi.Operation = .infer,
    model_family: []const u8,
    adapter_id: ?[]const u8 = null,
    model_name: ?[]const u8 = null,
    execution: ExecutionMode = .sync,
    allows_batching: bool = true,
    priority: u8 = 0,
};

pub const InputPayload = union(enum) {
    none,
    text: []const u8,
    image_path: []const u8,
    document_path: []const u8,
    audio_path: []const u8,
    video_path: []const u8,

    pub fn asString(self: InputPayload) ?[]const u8 {
        return switch (self) {
            .none => null,
            .text => |value| value,
            .image_path => |value| value,
            .document_path => |value| value,
            .audio_path => |value| value,
            .video_path => |value| value,
        };
    }
};

pub const GenerationOptions = struct {
    max_tokens: ?usize = null,
    native_execution: bool = false,
};

pub const TaskRequest = struct {
    spec: TaskSpec,
    input: InputPayload = .none,
    generation: GenerationOptions = .{},
};

test "task spec defaults are stable" {
    const spec = TaskSpec{
        .modality = .text,
        .operation = "generate",
        .operation_id = .generate,
        .model_family = "qwen3",
    };

    try std.testing.expectEqual(abi.Operation.generate, spec.operation_id);
    try std.testing.expectEqual(ExecutionMode.sync, spec.execution);
    try std.testing.expect(spec.allows_batching);
    try std.testing.expectEqual(@as(u8, 0), spec.priority);
}

test "task request carries typed payload and generation options" {
    const request = TaskRequest{
        .spec = .{
            .modality = .text,
            .operation = "generate",
            .operation_id = .generate,
            .model_family = "qwen3",
        },
        .input = .{ .text = "hello" },
        .generation = .{ .max_tokens = 128 },
    };

    try std.testing.expectEqualStrings("hello", request.input.asString().?);
    try std.testing.expectEqual(@as(?usize, 128), request.generation.max_tokens);
    try std.testing.expect(!request.generation.native_execution);
}
