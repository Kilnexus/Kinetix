const std = @import("std");
const engine = @import("engine_root");

const backend = engine.artifacts.backend;
const runtime_model = engine.runtime.model;
const runtime_session = engine.runtime.session;
const runtime_types = engine.runtime.types;
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
    unified_runtime: runtime_session.RuntimeSession,
    model_handle: runtime_model.ModelHandle,
    descriptor: runtime_types.Descriptor,

    pub fn deinit(self: *ExecutionContext) void {
        self.model_handle.deinit();
        self.unified_runtime.deinit();
        self.* = undefined;
    }

    pub fn prepare(self: *const ExecutionContext, request: ContextRequest) !PreparedContextExecution {
        const task_request = try buildTaskRequest(self.descriptor, request);
        const runtime_plan = try self.unified_runtime.plan(&self.model_handle, buildRuntimeRequest(self.descriptor, request));

        return .{
            .context = self,
            .request = task_request,
            .runtime_plan = runtime_plan,
        };
    }

    pub fn prepareBatch(self: *const ExecutionContext, allocator: std.mem.Allocator, request: ContextBatchRequest) !PreparedContextBatchExecution {
        const requests = try buildTaskRequests(allocator, self.descriptor, request.items);
        errdefer allocator.free(requests);
        var runtime_batch_request = try buildRuntimeBatchRequest(allocator, self.descriptor, request.items);
        defer runtime_batch_request.deinit();
        const runtime_plan = try self.unified_runtime.planBatch(&self.model_handle, runtime_batch_request.request);

        return .{
            .allocator = allocator,
            .context = self,
            .requests = requests,
            .runtime_plan = runtime_plan,
        };
    }

    pub fn execute(self: *const ExecutionContext, request: ContextRequest) !runtime_types.ExecutionResult {
        var prepared = try self.prepare(request);
        defer prepared.deinit();
        return try prepared.execute();
    }

    pub fn executeBatch(self: *const ExecutionContext, allocator: std.mem.Allocator, request: ContextBatchRequest) !runtime_types.BatchExecutionReport {
        var prepared = try self.prepareBatch(allocator, request);
        defer prepared.deinit();
        return try prepared.execute();
    }
};

pub const PreparedContextExecution = struct {
    context: *const ExecutionContext,
    request: task.TaskRequest,
    runtime_plan: runtime_types.ExecutionPlan,

    pub fn deinit(self: *PreparedContextExecution) void {
        self.runtime_plan.deinit();
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedContextExecution) !runtime_types.ExecutionResult {
        return try executeWithUnifiedRuntime(self.context, &self.runtime_plan, self.request.spec.execution);
    }
};

pub const PreparedContextBatchExecution = struct {
    allocator: std.mem.Allocator,
    context: *const ExecutionContext,
    requests: []task.TaskRequest,
    runtime_plan: runtime_types.ExecutionPlan,

    pub fn deinit(self: *PreparedContextBatchExecution) void {
        self.runtime_plan.deinit();
        self.allocator.free(self.requests);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedContextBatchExecution) !runtime_types.BatchExecutionReport {
        return try executeBatchWithUnifiedRuntime(self.context, &self.runtime_plan);
    }
};

pub const PreparedExecution = struct {
    allocator: std.mem.Allocator,
    context: *ExecutionContext,
    descriptor: runtime_types.Descriptor,
    request: task.TaskRequest,
    runtime_plan: runtime_types.ExecutionPlan,

    pub fn deinit(self: *PreparedExecution) void {
        self.runtime_plan.deinit();
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedExecution) !runtime_types.ExecutionResult {
        return try executeWithUnifiedRuntime(self.context, &self.runtime_plan, self.request.spec.execution);
    }
};

pub const PreparedBatchExecution = struct {
    allocator: std.mem.Allocator,
    context: *ExecutionContext,
    descriptor: runtime_types.Descriptor,
    requests: []task.TaskRequest,
    runtime_plan: runtime_types.ExecutionPlan,

    pub fn deinit(self: *PreparedBatchExecution) void {
        self.runtime_plan.deinit();
        self.allocator.free(self.requests);
        self.context.deinit();
        self.allocator.destroy(self.context);
        self.* = undefined;
    }

    pub fn execute(self: *const PreparedBatchExecution) !runtime_types.BatchExecutionReport {
        return try executeBatchWithUnifiedRuntime(self.context, &self.runtime_plan);
    }
};

pub fn openContext(allocator: std.mem.Allocator, request: OpenContextRequest) !*ExecutionContext {
    var unified = runtime_session.RuntimeSession.init(allocator);
    errdefer unified.deinit();

    var model_handle = try unified.openModel(.{
        .model_dir = request.model_dir,
        .preferred_weights = request.preferred_weights,
    });
    errdefer model_handle.deinit();

    const context = try allocator.create(ExecutionContext);
    errdefer allocator.destroy(context);
    context.* = .{
        .allocator = allocator,
        .unified_runtime = unified,
        .model_handle = model_handle,
        .descriptor = buildDescriptor(&model_handle),
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
        .runtime_plan = prepared.runtime_plan,
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
        .runtime_plan = prepared.runtime_plan,
    };
}

fn buildDescriptor(handle: *const runtime_model.ModelHandle) runtime_types.Descriptor {
    return .{
        .id = handle.normalized.descriptor.id,
        .modality = handle.normalized.descriptor.modality,
        .bound_model_family = handle.normalized.descriptor.family,
        .supports_batching = handle.normalized.capabilities.supports_batch,
        .supports_streaming = handle.normalized.capabilities.supports_stream,
        .supported_operations = handle.normalized.capabilities.supported_operations,
    };
}

fn defaultOperation(descriptor: runtime_types.Descriptor) []const u8 {
    if (descriptor.supported_operations.len == 0) return "infer";
    return descriptor.supported_operations[0];
}

const ResolvedRequest = struct {
    operation: []const u8,
    input: task.InputPayload,
    execution: task.ExecutionMode,
    generation: task.GenerationOptions,
    allows_batching: bool = true,
};

fn buildTaskRequest(descriptor: runtime_types.Descriptor, request: ContextRequest) !task.TaskRequest {
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;
    const resolved = resolveContextRequest(descriptor, request);

    return .{
        .spec = .{
            .modality = descriptor.modality,
            .operation = resolved.operation,
            .model_family = model_family,
            .adapter_id = descriptor.id,
            .execution = resolved.execution,
        },
        .input = resolved.input,
        .generation = resolved.generation,
    };
}

fn buildTaskRequests(allocator: std.mem.Allocator, descriptor: runtime_types.Descriptor, items: []const ContextBatchItem) ![]task.TaskRequest {
    const requests = try allocator.alloc(task.TaskRequest, items.len);
    errdefer allocator.free(requests);

    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;

    for (items, requests) |item, *slot| {
        const resolved = resolveContextBatchItem(descriptor, item);
        slot.* = .{
            .spec = .{
                .modality = descriptor.modality,
                .operation = resolved.operation,
                .model_family = model_family,
                .adapter_id = descriptor.id,
                .execution = resolved.execution,
                .allows_batching = resolved.allows_batching,
            },
            .input = resolved.input,
            .generation = resolved.generation,
        };
    }

    return requests;
}

const OwnedRuntimeBatchRequest = struct {
    allocator: std.mem.Allocator,
    request: runtime_types.RuntimeBatchRequest,

    pub fn deinit(self: *OwnedRuntimeBatchRequest) void {
        self.allocator.free(@constCast(self.request.items));
        self.* = undefined;
    }
};

fn buildRuntimeRequest(descriptor: runtime_types.Descriptor, request: ContextRequest) runtime_types.RuntimeRequest {
    const resolved = resolveContextRequest(descriptor, request);
    return .{
        .operation = resolved.operation,
        .input = resolved.input,
        .execution = resolved.execution,
        .generation = resolved.generation,
        .allows_batching = true,
    };
}

fn buildRuntimeBatchRequest(
    allocator: std.mem.Allocator,
    descriptor: runtime_types.Descriptor,
    items: []const ContextBatchItem,
) !OwnedRuntimeBatchRequest {
    const runtime_items = try allocator.alloc(runtime_types.RuntimeRequest, items.len);
    errdefer allocator.free(runtime_items);

    for (items, runtime_items) |item, *slot| {
        const resolved = resolveContextBatchItem(descriptor, item);
        slot.* = .{
            .operation = resolved.operation,
            .input = resolved.input,
            .execution = resolved.execution,
            .generation = resolved.generation,
            .allows_batching = resolved.allows_batching,
        };
    }

    return .{
        .allocator = allocator,
        .request = .{ .items = runtime_items },
    };
}

fn resolveContextRequest(descriptor: runtime_types.Descriptor, request: ContextRequest) ResolvedRequest {
    return .{
        .operation = request.operation orelse defaultOperation(descriptor),
        .input = inferInputPayload(descriptor.modality, request.input),
        .execution = request.execution,
        .generation = .{
            .max_tokens = request.max_tokens,
            .native_execution = request.native_exec,
        },
    };
}

fn resolveContextBatchItem(descriptor: runtime_types.Descriptor, item: ContextBatchItem) ResolvedRequest {
    return .{
        .operation = item.operation orelse defaultOperation(descriptor),
        .input = inferInputPayload(descriptor.modality, item.input),
        .execution = item.execution orelse .sync,
        .generation = .{
            .max_tokens = item.max_tokens,
            .native_execution = item.native_exec,
        },
        .allows_batching = item.allows_batching,
    };
}

fn inferInputPayload(modality: task.Modality, input: ?[]const u8) task.InputPayload {
    const value = input orelse return .none;
    return switch (modality) {
        .text, .tts, .multimodal => .{ .text = value },
        .vision => .{ .image_path = value },
        .ocr => if (isDocumentPath(value)) .{ .document_path = value } else .{ .image_path = value },
        .audio => .{ .audio_path = value },
        .video => .{ .video_path = value },
    };
}

fn isDocumentPath(path: []const u8) bool {
    return std.mem.endsWith(u8, path, ".pdf") or
        std.mem.endsWith(u8, path, ".PDF");
}

fn executeWithUnifiedRuntime(
    context: *const ExecutionContext,
    runtime_plan: *const runtime_types.ExecutionPlan,
    execution: task.ExecutionMode,
) !runtime_types.ExecutionResult {
    var runtime_result = try context.unified_runtime.execute(&context.model_handle, runtime_plan);

    const output = runtime_result.output;
    runtime_result.output = .none;
    return .{
        .submission = .{
            .adapter_id = context.descriptor.id,
            .accepted = true,
            .execution = execution,
        },
        .origin = runtime_result.origin,
        .note = runtime_result.note,
        .output = output,
    };
}

fn executeBatchWithUnifiedRuntime(
    context: *const ExecutionContext,
    runtime_plan: *const runtime_types.ExecutionPlan,
) !runtime_types.BatchExecutionReport {
    var runtime_results = try context.unified_runtime.executeBatch(&context.model_handle, runtime_plan);
    defer runtime_results.deinit();

    const batches = try context.allocator.alloc(runtime_types.ExecutedBatch, runtime_plan.batches.len);
    errdefer context.allocator.free(batches);

    var initialized_batches: usize = 0;
    errdefer {
        for (batches[0..initialized_batches]) |batch| {
            for (batch.request_results) |*request_result| request_result.result.deinit(context.allocator);
            context.allocator.free(batch.request_results);
        }
        context.allocator.free(batches);
    }

    for (runtime_plan.batches, batches) |plan_batch, *batch| {
        const request_results = try context.allocator.alloc(runtime_types.RequestExecutionResult, plan_batch.request_indices.len);
        errdefer context.allocator.free(request_results);

        for (plan_batch.request_indices, request_results) |request_index, *request_result| {
            if (request_index >= runtime_results.items.len) return error.InvalidExecutionPlan;

            const runtime_result = &runtime_results.items[request_index];
            const output = runtime_result.output;
            runtime_result.output = .none;
            request_result.* = .{
                .request_index = request_index,
                .result = .{
                    .submission = .{
                        .adapter_id = context.descriptor.id,
                        .accepted = true,
                        .execution = runtime_plan.requests[request_index].execution,
                    },
                    .origin = runtime_result.origin,
                    .note = runtime_result.note,
                    .output = output,
                },
            };
        }

        batch.* = .{
            .adapter_id = context.descriptor.id,
            .execution = plan_batch.execution,
            .supports_batching = plan_batch.allows_batching,
            .execute_path = if (plan_batch.allows_batching and plan_batch.request_indices.len > 1)
                .adapter_batch
            else
                .per_request_fallback,
            .request_results = request_results,
        };
        initialized_batches += 1;
    }

    return .{
        .allocator = context.allocator,
        .batches = batches,
    };
}

test "prepared execution resolves text requests through the unified runtime" {
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
    try std.testing.expectEqual(runtime_types.ExecutionPath.stream, prepared.runtime_plan.path);
    try std.testing.expect(prepared.descriptor.supports_streaming);

    var result = try prepared.execute();
    defer result.deinit(std.testing.allocator);
    try std.testing.expect(result.submission.accepted);
    try std.testing.expectEqual(runtime_types.ExecutionNote.text_request_ready, result.note);
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

    try std.testing.expectEqual(runtime_types.ExecutionPath.native_accelerated, prepared.runtime_plan.path);
    try std.testing.expectEqual(runtime_types.ExecutionOrigin.native_single, result.origin);
    try std.testing.expectEqual(runtime_types.ExecutionNote.text_native_qwen_single, result.note);
    try std.testing.expectEqualStrings("test-native-single", result.output.text);
}

test "prepared batch execution groups batchable text requests in the unified planner" {
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
    try std.testing.expectEqual(@as(usize, 2), prepared.runtime_plan.batches.len);
    try std.testing.expectEqual(@as(usize, 2), prepared.runtime_plan.batches[0].request_indices.len);
    try std.testing.expectEqual(@as(usize, 1), prepared.runtime_plan.batches[1].request_indices.len);

    var report = try prepared.execute();
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 3), report.totalRequests());
    try std.testing.expectEqual(@as(usize, 3), report.totalAccepted());
    try std.testing.expectEqual(@as(usize, 2), report.batches.len);
}

test "execution context reuses one opened runtime handle across multiple text requests" {
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

    try std.testing.expectEqual(runtime_types.ProviderKey.qwen3_text, context.model_handle.normalized.provider_key);
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

    try std.testing.expectEqual(runtime_types.ExecutionOrigin.native_single, first.origin);
    try std.testing.expectEqual(runtime_types.ExecutionOrigin.native_single, second.origin);
    try std.testing.expectEqualStrings("test-native-single", first.output.text);
    try std.testing.expectEqualStrings("test-native-single", second.output.text);
}

test "prepared execution resolves tts requests through the unified runtime" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeMossTtsBundle(tmp.dir);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var prepared = try prepare(std.testing.allocator, .{
        .model_dir = root_path,
        .operation = "synthesize",
        .input = "hello moss",
    });
    defer prepared.deinit();

    try std.testing.expectEqual(task.Modality.tts, prepared.descriptor.modality);
    try std.testing.expectEqualStrings("moss_tts_nano", prepared.request.spec.model_family);
    try std.testing.expectEqualStrings("hello moss", prepared.request.input.asString().?);
    try std.testing.expectEqual(runtime_types.ExecutionPath.runtime_backend, prepared.runtime_plan.path);

    var result = try prepared.execute();
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(runtime_types.ExecutionNote.tts_model_ready, result.note);
    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"provider_key\":\"moss_tts_nano_tts\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"output_contract\":\"audio_path\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"input_text\":\"hello moss\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"normalized_text\":\"hello moss.\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"chunk_count\":1") != null);
}

test "ocr input inference maps pdf paths to document payloads" {
    const payload = inferInputPayload(.ocr, "demo.pdf");
    try std.testing.expectEqual(task.InputPayload.document_path, std.meta.activeTag(payload));
    try std.testing.expectEqualStrings("demo.pdf", payload.asString().?);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn writeMossTtsBundle(dir: std.fs.Dir) !void {
    try dir.makeDir("MOSS-TTS-Nano-100M-ONNX");
    try dir.makeDir("MOSS-Audio-Tokenizer-Nano-ONNX");

    var tts_dir = try dir.openDir("MOSS-TTS-Nano-100M-ONNX", .{});
    defer tts_dir.close();
    var codec_dir = try dir.openDir("MOSS-Audio-Tokenizer-Nano-ONNX", .{});
    defer codec_dir.close();

    try writeTmpFile(tts_dir, "browser_poc_manifest.json",
        \\{
        \\  "builtin_voices": ["speaker_a", "speaker_b"],
        \\  "model_files": {}
        \\}
    );
    try writeTmpFile(tts_dir, "tts_browser_onnx_meta.json",
        \\{
        \\  "model_info": {
        \\    "name": "MOSS-TTS-Nano-100M"
        \\  }
        \\}
    );
    try writeTmpFile(tts_dir, "tokenizer.model", "synthetic sentencepiece model");
    try writeTmpFile(codec_dir, "codec_browser_onnx_meta.json",
        \\{
        \\  "codec_config": {
        \\    "sample_rate": 48000,
        \\    "channels": 2,
        \\    "num_quantizers": 32
        \\  }
        \\}
    );
}
