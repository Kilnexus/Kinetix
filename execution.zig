const std = @import("std");
const adapters = @import("adapters/adapters.zig");
const engine = @import("engine/kinetix.zig");

const backend = engine.artifacts.backend;
const legacy_command = adapters.legacy_command;
const registry_mod = engine.registry;
const scheduler_mod = engine.scheduler;
const task = engine.core.task;

pub const PrepareRequest = struct {
    model_dir: []const u8,
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
    max_tokens: ?usize = null,
};

pub const PrepareBatchItem = struct {
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: ?task.ExecutionMode = null,
    max_tokens: ?usize = null,
    allows_batching: bool = true,
};

pub const PrepareBatchRequest = struct {
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme = .auto,
    items: []const PrepareBatchItem,
};

pub const LegacyOptions = struct {
    input: ?[]const u8 = null,
    max_tokens: ?usize = null,
};

pub const PreparedExecution = struct {
    allocator: std.mem.Allocator,
    registry: registry_mod.Registry,
    managed: adapters.factory.ManagedAdapter,
    descriptor: engine.adapter.Descriptor,
    request: task.TaskRequest,
    plan: scheduler_mod.DispatchPlan,
    submission: engine.adapter.Submission,

    pub fn deinit(self: *PreparedExecution) void {
        self.managed.deinit();
        self.registry.deinit();
        self.* = undefined;
    }

    pub fn prepareLegacyCommand(self: *const PreparedExecution, options: LegacyOptions) !legacy_command.LegacyCommand {
        const resolved_input = options.input orelse self.request.input.asString();
        const resolved_max_tokens = options.max_tokens orelse self.request.generation.max_tokens;
        return try self.managed.buildLegacyCommand(self.allocator, .{
            .operation = self.request.spec.operation,
            .input = resolved_input,
            .max_tokens = resolved_max_tokens,
        });
    }

    pub fn executeLegacy(self: *const PreparedExecution, options: LegacyOptions) !std.process.Child.Term {
        var command = try self.prepareLegacyCommand(options);
        defer command.deinit();
        return try executeLegacyCommand(command);
    }
};

pub const PreparedBatchExecution = struct {
    allocator: std.mem.Allocator,
    registry: registry_mod.Registry,
    managed: adapters.factory.ManagedAdapter,
    descriptor: engine.adapter.Descriptor,
    requests: []task.TaskRequest,
    batch_plan: scheduler_mod.BatchPlan,

    pub fn deinit(self: *PreparedBatchExecution) void {
        self.batch_plan.deinit();
        self.allocator.free(self.requests);
        self.managed.deinit();
        self.registry.deinit();
        self.* = undefined;
    }
};

pub fn prepare(allocator: std.mem.Allocator, request: PrepareRequest) !PreparedExecution {
    var registry = registry_mod.Registry.init(allocator);
    errdefer registry.deinit();

    var managed = try adapters.factory.initAuto(allocator, request.model_dir, request.preferred_weights);
    errdefer managed.deinit();
    try managed.registerInto(&registry);

    const descriptor = managed.descriptor();
    const operation = request.operation orelse defaultOperation(descriptor);
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;
    const scheduler = scheduler_mod.Scheduler.init(&registry);

    const spec = task.TaskSpec{
        .modality = descriptor.modality,
        .operation = operation,
        .model_family = model_family,
        .adapter_id = descriptor.id,
        .execution = request.execution,
    };
    const task_request = task.TaskRequest{
        .spec = spec,
        .input = inferInputPayload(descriptor.modality, request.input),
        .generation = .{ .max_tokens = request.max_tokens },
    };

    return .{
        .allocator = allocator,
        .registry = registry,
        .managed = managed,
        .descriptor = descriptor,
        .request = task_request,
        .plan = try scheduler.planRequest(task_request),
        .submission = try scheduler.submit(task_request),
    };
}

pub fn prepareBatch(allocator: std.mem.Allocator, request: PrepareBatchRequest) !PreparedBatchExecution {
    var registry = registry_mod.Registry.init(allocator);
    errdefer registry.deinit();

    var managed = try adapters.factory.initAuto(allocator, request.model_dir, request.preferred_weights);
    errdefer managed.deinit();
    try managed.registerInto(&registry);

    const descriptor = managed.descriptor();
    const operation = defaultOperation(descriptor);
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;
    const scheduler = scheduler_mod.Scheduler.init(&registry);

    const requests = try allocator.alloc(task.TaskRequest, request.items.len);
    errdefer allocator.free(requests);

    for (request.items, requests) |item, *slot| {
        const execution_mode = item.execution orelse .sync;
        slot.* = .{
            .spec = .{
                .modality = descriptor.modality,
                .operation = item.operation orelse operation,
                .model_family = model_family,
                .adapter_id = descriptor.id,
                .execution = execution_mode,
                .allows_batching = item.allows_batching,
            },
            .input = inferInputPayload(descriptor.modality, item.input),
            .generation = .{ .max_tokens = item.max_tokens },
        };
    }

    return .{
        .allocator = allocator,
        .registry = registry,
        .managed = managed,
        .descriptor = descriptor,
        .requests = requests,
        .batch_plan = try scheduler.planBatches(allocator, requests),
    };
}

pub fn executeLegacyCommand(command: legacy_command.LegacyCommand) !std.process.Child.Term {
    var child = std.process.Child.init(command.argv, std.heap.page_allocator);
    child.cwd = command.workdir;
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    try child.spawn();
    return try child.wait();
}

fn defaultOperation(descriptor: engine.adapter.Descriptor) []const u8 {
    if (descriptor.supported_operations.len == 0) return "infer";
    return descriptor.supported_operations[0];
}

fn inferInputPayload(modality: task.Modality, input: ?[]const u8) task.InputPayload {
    const value = input orelse return .none;
    return switch (modality) {
        .text, .multimodal => .{ .text = value },
        .vision, .ocr => .{ .image_path = value },
        .audio, .tts => .{ .audio_path = value },
        .video => .{ .video_path = value },
    };
}

test "prepared execution resolves text adapter through shared executor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var prepared = try prepare(std.testing.allocator, .{
        .model_dir = root_path,
        .input = "hello",
        .execution = .stream,
        .max_tokens = 32,
    });
    defer prepared.deinit();

    try std.testing.expectEqualStrings("qwen3", prepared.request.spec.model_family);
    try std.testing.expectEqualStrings("hello", prepared.request.input.asString().?);
    try std.testing.expectEqual(@as(?usize, 32), prepared.request.generation.max_tokens);
    try std.testing.expectEqual(task.ExecutionMode.stream, prepared.plan.execution);
    try std.testing.expect(prepared.plan.supports_streaming);
}

test "prepared batch execution groups compatible text requests" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const items = [_]PrepareBatchItem{
        .{ .input = "hello" },
        .{ .input = "world" },
        .{ .input = "solo", .allows_batching = false },
    };

    var prepared = try prepareBatch(std.testing.allocator, .{
        .model_dir = root_path,
        .items = &items,
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 3), prepared.requests.len);
    try std.testing.expectEqual(@as(usize, 2), prepared.batch_plan.batches.len);
    try std.testing.expectEqual(@as(usize, 2), prepared.batch_plan.batches[0].len());
    try std.testing.expectEqual(@as(usize, 1), prepared.batch_plan.batches[1].len());
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
