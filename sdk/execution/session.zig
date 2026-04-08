const std = @import("std");
const adapters = @import("adapters_root");
const engine = @import("engine_root");

const backend = engine.artifacts.backend;
const registry_mod = engine.registry;
const batch_executor = engine.runtime.batch_executor;
const scheduler_mod = engine.scheduler;
const task = engine.core.task;

pub const PrepareRequest = struct {
    model_dir: []const u8,
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const PrepareBatchItem = struct {
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: ?task.ExecutionMode = null,
    max_tokens: ?usize = null,
    native_exec: bool = false,
    allows_batching: bool = true,
};

pub const PrepareBatchRequest = struct {
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme = .auto,
    items: []const PrepareBatchItem,
};

pub const OpenContextRequest = struct {
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme = .auto,
};

pub const ContextRequest = struct {
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const ContextBatchItem = struct {
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: ?task.ExecutionMode = null,
    max_tokens: ?usize = null,
    native_exec: bool = false,
    allows_batching: bool = true,
};

pub const ContextBatchRequest = struct {
    items: []const ContextBatchItem,
};

pub const ExecutionContext = struct {
    allocator: std.mem.Allocator,
    registry: registry_mod.Registry,
    managed: *adapters.factory.ManagedAdapter,
    descriptor: engine.adapter.Descriptor,

    pub fn deinit(self: *ExecutionContext) void {
        self.managed.deinit();
        self.allocator.destroy(self.managed);
        self.registry.deinit();
        self.* = undefined;
    }

    pub fn prepare(self: *const ExecutionContext, request: ContextRequest) !PreparedContextExecution {
        const scheduler = scheduler_mod.Scheduler.init(&self.registry);
        const task_request = buildTaskRequest(self.descriptor, request);

        return .{
            .context = self,
            .request = task_request,
            .plan = try scheduler.planRequest(task_request),
            .submission = try scheduler.submit(task_request),
        };
    }

    pub fn prepareBatch(self: *const ExecutionContext, allocator: std.mem.Allocator, request: ContextBatchRequest) !PreparedContextBatchExecution {
        const scheduler = scheduler_mod.Scheduler.init(&self.registry);
        const requests = try buildTaskRequests(allocator, self.descriptor, request.items);
        errdefer allocator.free(requests);

        return .{
            .allocator = allocator,
            .context = self,
            .requests = requests,
            .batch_plan = try scheduler.planBatches(allocator, requests),
        };
    }

    pub fn execute(self: *const ExecutionContext, request: ContextRequest) !engine.adapter.ExecutionResult {
        var prepared = try self.prepare(request);
        defer prepared.deinit();
        return try prepared.execute();
    }

    pub fn executeBatch(self: *const ExecutionContext, allocator: std.mem.Allocator, request: ContextBatchRequest) !batch_executor.BatchExecutionReport {
        var prepared = try self.prepareBatch(allocator, request);
        defer prepared.deinit();
        return try prepared.execute();
    }
};

pub const PreparedContextExecution = struct {
    context: *const ExecutionContext,
    request: task.TaskRequest,
    plan: scheduler_mod.DispatchPlan,
    submission: engine.adapter.Submission,

    pub fn deinit(self: *PreparedContextExecution) void {
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedContextExecution) !engine.adapter.ExecutionResult {
        const entry = self.context.registry.findById(self.plan.adapter_id) orelse return error.AdapterNotFound;
        return try entry.adapter.execute(self.context.allocator, self.request);
    }
};

pub const PreparedContextBatchExecution = struct {
    allocator: std.mem.Allocator,
    context: *const ExecutionContext,
    requests: []task.TaskRequest,
    batch_plan: scheduler_mod.BatchPlan,

    pub fn deinit(self: *PreparedContextBatchExecution) void {
        self.batch_plan.deinit();
        self.allocator.free(self.requests);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedContextBatchExecution) !batch_executor.BatchExecutionReport {
        return try batch_executor.execute(self.allocator, &self.context.registry, self.requests, &self.batch_plan);
    }
};

pub const PreparedExecution = struct {
    allocator: std.mem.Allocator,
    context: *ExecutionContext,
    descriptor: engine.adapter.Descriptor,
    request: task.TaskRequest,
    plan: scheduler_mod.DispatchPlan,
    submission: engine.adapter.Submission,

    pub fn deinit(self: *PreparedExecution) void {
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedExecution) !engine.adapter.ExecutionResult {
        const entry = self.context.registry.findById(self.plan.adapter_id) orelse return error.AdapterNotFound;
        return try entry.adapter.execute(self.allocator, self.request);
    }
};

pub const PreparedBatchExecution = struct {
    allocator: std.mem.Allocator,
    context: *ExecutionContext,
    descriptor: engine.adapter.Descriptor,
    requests: []task.TaskRequest,
    batch_plan: scheduler_mod.BatchPlan,

    pub fn deinit(self: *PreparedBatchExecution) void {
        self.batch_plan.deinit();
        self.allocator.free(self.requests);
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedBatchExecution) !batch_executor.BatchExecutionReport {
        return try batch_executor.execute(self.allocator, &self.context.registry, self.requests, &self.batch_plan);
    }
};

pub fn openContext(allocator: std.mem.Allocator, request: OpenContextRequest) !*ExecutionContext {
    var registry = registry_mod.Registry.init(allocator);
    errdefer registry.deinit();

    const managed = try allocator.create(adapters.factory.ManagedAdapter);
    errdefer allocator.destroy(managed);
    managed.* = try adapters.factory.initAuto(allocator, request.model_dir, request.preferred_weights);
    errdefer managed.deinit();
    try managed.registerInto(&registry);

    const context = try allocator.create(ExecutionContext);
    errdefer allocator.destroy(context);
    context.* = .{
        .allocator = allocator,
        .registry = registry,
        .managed = managed,
        .descriptor = managed.descriptor(),
    };
    return context;
}

pub fn prepare(allocator: std.mem.Allocator, request: PrepareRequest) !PreparedExecution {
    const context = try openContext(allocator, .{
        .model_dir = request.model_dir,
        .preferred_weights = request.preferred_weights,
    });
    errdefer {
        context.deinit();
        allocator.destroy(context);
    }

    const prepared = try context.prepare(.{
        .operation = request.operation,
        .input = request.input,
        .execution = request.execution,
        .max_tokens = request.max_tokens,
        .native_exec = request.native_exec,
    });

    return .{
        .allocator = allocator,
        .context = context,
        .descriptor = context.descriptor,
        .request = prepared.request,
        .plan = prepared.plan,
        .submission = prepared.submission,
    };
}

pub fn prepareBatch(allocator: std.mem.Allocator, request: PrepareBatchRequest) !PreparedBatchExecution {
    const context = try openContext(allocator, .{
        .model_dir = request.model_dir,
        .preferred_weights = request.preferred_weights,
    });
    errdefer {
        context.deinit();
        allocator.destroy(context);
    }

    const batch_items = try allocator.alloc(ContextBatchItem, request.items.len);
    defer allocator.free(batch_items);

    for (request.items, batch_items) |item, *slot| {
        slot.* = .{
            .operation = item.operation,
            .input = item.input,
            .execution = item.execution,
            .max_tokens = item.max_tokens,
            .native_exec = item.native_exec,
            .allows_batching = item.allows_batching,
        };
    }

    const prepared = try context.prepareBatch(allocator, .{ .items = batch_items });

    return .{
        .allocator = allocator,
        .context = context,
        .descriptor = context.descriptor,
        .requests = prepared.requests,
        .batch_plan = prepared.batch_plan,
    };
}

fn defaultOperation(descriptor: engine.adapter.Descriptor) []const u8 {
    if (descriptor.supported_operations.len == 0) return "infer";
    return descriptor.supported_operations[0];
}

fn buildTaskRequest(descriptor: engine.adapter.Descriptor, request: ContextRequest) !task.TaskRequest {
    const operation = request.operation orelse defaultOperation(descriptor);
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;

    return .{
        .spec = .{
            .modality = descriptor.modality,
            .operation = operation,
            .model_family = model_family,
            .adapter_id = descriptor.id,
            .execution = request.execution,
        },
        .input = inferInputPayload(descriptor.modality, request.input),
        .generation = .{
            .max_tokens = request.max_tokens,
            .native_execution = request.native_exec,
        },
    };
}

fn buildTaskRequests(allocator: std.mem.Allocator, descriptor: engine.adapter.Descriptor, items: []const ContextBatchItem) ![]task.TaskRequest {
    const requests = try allocator.alloc(task.TaskRequest, items.len);
    errdefer allocator.free(requests);

    const operation = defaultOperation(descriptor);
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;

    for (items, requests) |item, *slot| {
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
            .generation = .{
                .max_tokens = item.max_tokens,
                .native_execution = item.native_exec,
            },
        };
    }

    return requests;
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

    var result = try prepared.execute();
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(result.submission.accepted);
    try std.testing.expectEqual(engine.adapter.ExecutionNote.text_request_ready, result.note);
}

test "prepared execution can request native text execution output" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var prepared = try prepare(std.testing.allocator, .{
        .model_dir = root_path,
        .operation = "generate",
        .input = "hello",
        .max_tokens = 8,
        .native_exec = true,
    });
    defer prepared.deinit();

    var result = try prepared.execute();
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(engine.adapter.ExecutionOrigin.native_single_bridge, result.origin);
    try std.testing.expectEqual(engine.adapter.ExecutionNote.text_native_qwen_single, result.note);
    try std.testing.expectEqualStrings("stub-native-single", result.output.text);
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

    var report = try prepared.execute();
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 3), report.totalRequests());
    try std.testing.expectEqual(@as(usize, 3), report.totalAccepted());
    try std.testing.expectEqual(@as(usize, 2), report.batches.len);
}

test "execution context reuses one opened adapter across multiple text requests" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const context = try openContext(std.testing.allocator, .{
        .model_dir = root_path,
    });
    defer {
        context.deinit();
        std.testing.allocator.destroy(context);
    }

    try std.testing.expectEqual(@as(usize, 1), context.registry.count());
    try std.testing.expectEqualStrings("qwen3", context.descriptor.bound_model_family.?);

    var first = try context.execute(.{
        .operation = "generate",
        .input = "hello",
        .max_tokens = 8,
        .native_exec = true,
    });
    defer first.deinit(std.testing.allocator);

    var second = try context.execute(.{
        .operation = "chat",
        .input = "hello again",
        .max_tokens = 8,
        .native_exec = true,
    });
    defer second.deinit(std.testing.allocator);

    try std.testing.expectEqual(engine.adapter.ExecutionOrigin.native_single_bridge, first.origin);
    try std.testing.expectEqual(engine.adapter.ExecutionOrigin.native_single_bridge, second.origin);
    try std.testing.expectEqualStrings("stub-native-single", first.output.text);
    try std.testing.expectEqualStrings("stub-native-single", second.output.text);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
