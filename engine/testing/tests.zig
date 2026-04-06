const std = @import("std");
const task = @import("../core/task.zig");
const adapter_mod = @import("../adapter/adapter.zig");
const backend = @import("../artifacts/backend/backend.zig");
const graph = @import("../artifacts/graph/graph.zig");
const memory = @import("../core/memory/memory.zig");
const registry_mod = @import("../registry/registry.zig");
const batch_executor_mod = @import("../runtime/batch_executor.zig");
const load_plan = @import("../runtime/load_plan.zig");
const scheduler_mod = @import("../scheduler/scheduler.zig");
const adapter_factory_mod = @import("../../adapters/factory.zig");
const ocr_adapter_mod = @import("../../adapters/ocr/ocr.zig");
const text_adapter_mod = @import("../../adapters/text/text.zig");
const vision_adapter_mod = @import("../../adapters/vision/vision.zig");

const MockState = struct {
    adapter_id: []const u8,
    submit_count: usize = 0,
    submit_batch_count: usize = 0,
};

fn submitMock(ctx: *anyopaque, request: task.TaskRequest) !adapter_mod.Submission {
    const state: *MockState = @ptrCast(@alignCast(ctx));
    state.submit_count += 1;
    return .{
        .adapter_id = state.adapter_id,
        .accepted = true,
        .execution = request.spec.execution,
    };
}

fn submitMockBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.Submission {
    const state: *MockState = @ptrCast(@alignCast(ctx));
    state.submit_batch_count += 1;

    const submissions = try allocator.alloc(adapter_mod.Submission, requests.len);
    errdefer allocator.free(submissions);

    for (requests, submissions) |request, *submission| {
        submission.* = .{
            .adapter_id = state.adapter_id,
            .accepted = true,
            .execution = request.spec.execution,
        };
    }

    return submissions;
}

const mock_vtable = adapter_mod.VTable{
    .submit = submitMock,
    .submit_batch = submitMockBatch,
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
            .bound_model_family = "qwen3",
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
            .bound_model_family = "qwen3",
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

    const submission = try scheduler.submit(.{ .spec = spec });
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

test "scheduler groups compatible text requests into one batch" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var state = MockState{ .adapter_id = "text.qwen3" };
    try registry.register(.{
        .ctx = &state,
        .descriptor = .{
            .id = "text.qwen3",
            .modality = .text,
            .bound_model_family = "qwen3",
            .supports_batching = true,
            .supports_streaming = true,
            .supported_operations = &.{ "generate" },
        },
        .vtable = &mock_vtable,
    });

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const requests = [_]task.TaskRequest{
        .{
            .spec = .{
                .modality = .text,
                .operation = "generate",
                .model_family = "qwen3",
                .execution = .sync,
            },
            .input = .{ .text = "hello" },
        },
        .{
            .spec = .{
                .modality = .text,
                .operation = "generate",
                .model_family = "qwen3",
                .execution = .sync,
            },
            .input = .{ .text = "world" },
        },
    };

    var batch_plan = try scheduler.planBatches(std.testing.allocator, &requests);
    defer batch_plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), batch_plan.batches.len);
    try std.testing.expectEqual(@as(usize, 2), batch_plan.batches[0].len());
    try std.testing.expect(batch_plan.batches[0].supports_batching);
}

test "scheduler keeps non-batchable OCR requests isolated" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var state = MockState{ .adapter_id = "ocr.swiftocr" };
    try registry.register(.{
        .ctx = &state,
        .descriptor = .{
            .id = "ocr.swiftocr",
            .modality = .ocr,
            .bound_model_family = "swiftocr",
            .supports_batching = false,
            .supported_operations = &.{ "infer-ocr" },
        },
        .vtable = &mock_vtable,
    });

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const requests = [_]task.TaskRequest{
        .{
            .spec = .{
                .modality = .ocr,
                .operation = "infer-ocr",
                .model_family = "swiftocr",
                .execution = .sync,
            },
            .input = .{ .image_path = "a.png" },
        },
        .{
            .spec = .{
                .modality = .ocr,
                .operation = "infer-ocr",
                .model_family = "swiftocr",
                .execution = .sync,
            },
            .input = .{ .image_path = "b.png" },
        },
    };

    var batch_plan = try scheduler.planBatches(std.testing.allocator, &requests);
    defer batch_plan.deinit();

    try std.testing.expectEqual(@as(usize, 2), batch_plan.batches.len);
    try std.testing.expect(!batch_plan.batches[0].supports_batching);
    try std.testing.expectEqual(@as(usize, 1), batch_plan.batches[0].len());
    try std.testing.expectEqual(@as(usize, 1), batch_plan.batches[1].len());
}

test "batch executor submits planned requests through shared adapter interface" {
    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();

    var state = MockState{ .adapter_id = "text.qwen3" };
    try registry.register(.{
        .ctx = &state,
        .descriptor = .{
            .id = "text.qwen3",
            .modality = .text,
            .bound_model_family = "qwen3",
            .supports_batching = true,
            .supports_streaming = true,
            .supported_operations = &.{ "generate" },
        },
        .vtable = &mock_vtable,
    });

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const requests = [_]task.TaskRequest{
        .{
            .spec = .{
                .modality = .text,
                .operation = "generate",
                .model_family = "qwen3",
                .execution = .sync,
            },
            .input = .{ .text = "hello" },
        },
        .{
            .spec = .{
                .modality = .text,
                .operation = "generate",
                .model_family = "qwen3",
                .execution = .sync,
            },
            .input = .{ .text = "world" },
        },
    };

    var batch_plan = try scheduler.planBatches(std.testing.allocator, &requests);
    defer batch_plan.deinit();

    var report = try batch_executor_mod.execute(std.testing.allocator, &registry, &requests, &batch_plan);
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 1), report.batches.len);
    try std.testing.expectEqual(@as(usize, 2), report.totalRequests());
    try std.testing.expectEqual(@as(usize, 2), report.totalAccepted());
    try std.testing.expectEqual(@as(usize, 0), state.submit_count);
    try std.testing.expectEqual(@as(usize, 1), state.submit_batch_count);
}

test "text adapter resolves qwen3 model and integrates with scheduler" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var text_adapter = try text_adapter_mod.TextAdapter.init(std.testing.allocator, root_path, .auto);
    defer text_adapter.deinit();

    try std.testing.expectEqualStrings("qwen3", text_adapter.descriptor.bound_model_family.?);
    try std.testing.expect(text_adapter.descriptor.supports_streaming);
    try std.testing.expect(text_adapter.descriptor.supportsOperation("generate"));

    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();
    try text_adapter.registerInto(&registry);

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const submission = try scheduler.submit(.{ .spec = .{
        .modality = .text,
        .operation = "generate",
        .model_family = "qwen3",
        .adapter_id = text_adapter.descriptor.id,
        .execution = .stream,
    } });

    try std.testing.expect(submission.accepted);
    try std.testing.expectEqual(task.ExecutionMode.stream, submission.execution);
}

test "text adapter restricts bert model to bert-compatible operations" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"bert\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.safetensors", "bf16");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var text_adapter = try text_adapter_mod.TextAdapter.init(std.testing.allocator, root_path, .auto);
    defer text_adapter.deinit();

    try std.testing.expectEqualStrings("bert", text_adapter.descriptor.bound_model_family.?);
    try std.testing.expect(!text_adapter.descriptor.supports_streaming);
    try std.testing.expect(text_adapter.descriptor.supportsOperation("fill-mask"));
    try std.testing.expect(!text_adapter.descriptor.supportsOperation("generate"));

    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();
    try text_adapter.registerInto(&registry);

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    try std.testing.expectError(error.NoMatchingAdapter, scheduler.submit(.{ .spec = .{
        .modality = .text,
        .operation = "generate",
        .model_family = "bert",
        .execution = .sync,
    } }));
}

test "vision adapter resolves yolo artifacts and integrates with scheduler" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json",
        \\{
        \\  "format_version": 1,
        \\  "model_name": "vision-yolo",
        \\  "metadata": { "class_count": 2 },
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "pipeline.detect", "kind": "Detect", "from": [-1] }
        \\  ],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "pipeline.detect",
        \\        "kind": "Detect",
        \\        "attrs": {},
        \\        "children": []
        \\      }
        \\    ]
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var vision_adapter = try vision_adapter_mod.VisionAdapter.init(std.testing.allocator, root_path);
    defer vision_adapter.deinit();

    try std.testing.expectEqualStrings("yolo", vision_adapter.descriptor.bound_model_family.?);
    try std.testing.expect(vision_adapter.descriptor.supportsOperation("detect"));
    try std.testing.expect(!vision_adapter.descriptor.supportsOperation("generate"));

    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();
    try vision_adapter.registerInto(&registry);

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const submission = try scheduler.submit(.{ .spec = .{
        .modality = .vision,
        .operation = "detect",
        .model_family = "yolo",
        .execution = .sync,
    } });

    try std.testing.expect(submission.accepted);
    try std.testing.expectEqual(task.ExecutionMode.sync, submission.execution);
}

test "vision adapter rejects missing binary weights" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json",
        \\{
        \\  "format_version": 1,
        \\  "model_name": "vision-generic",
        \\  "metadata": {},
        \\  "tensors": [],
        \\  "execution_plan": [],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": []
        \\  }
        \\}
    );

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    try std.testing.expectError(error.MissingBinaryWeightsArtifact, vision_adapter_mod.VisionAdapter.init(std.testing.allocator, root_path));
}

test "model catalog discovers swiftocr single-file model artifacts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "demo.swm", "SWOCR01");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try backend.ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    try std.testing.expect(catalog.has(.ocr_model));
}

test "ocr adapter resolves swiftocr model and integrates with scheduler" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "demo.swm", "SWOCR01");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var ocr_adapter = try ocr_adapter_mod.OCRAdapter.init(std.testing.allocator, root_path);
    defer ocr_adapter.deinit();

    try std.testing.expectEqualStrings("swiftocr", ocr_adapter.descriptor.bound_model_family.?);
    try std.testing.expect(ocr_adapter.descriptor.supportsOperation("infer-ocr"));
    try std.testing.expect(!ocr_adapter.descriptor.supportsOperation("generate"));

    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();
    try ocr_adapter.registerInto(&registry);

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    const submission = try scheduler.submit(.{ .spec = .{
        .modality = .ocr,
        .operation = "infer-ocr",
        .model_family = "swiftocr",
        .execution = .sync,
    } });

    try std.testing.expect(submission.accepted);
    try std.testing.expectEqual(task.ExecutionMode.sync, submission.execution);
}

test "text adapter rejects image payloads in shared request path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var text_adapter = try text_adapter_mod.TextAdapter.init(std.testing.allocator, root_path, .auto);
    defer text_adapter.deinit();

    var registry = registry_mod.Registry.init(std.testing.allocator);
    defer registry.deinit();
    try text_adapter.registerInto(&registry);

    const scheduler = scheduler_mod.Scheduler.init(&registry);
    try std.testing.expectError(error.InvalidInputPayload, scheduler.submit(.{
        .spec = .{
            .modality = .text,
            .operation = "generate",
            .model_family = "qwen3",
            .execution = .sync,
        },
        .input = .{ .image_path = "demo.png" },
    }));

}

test "adapter factory auto-detects text model directories" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var managed = try adapter_factory_mod.initAuto(std.testing.allocator, root_path, .auto);
    defer managed.deinit();

    try std.testing.expect(std.mem.startsWith(u8, managed.descriptorId(), "text.qwen3."));
}

test "adapter factory auto-detects vision model directories" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json",
        \\{
        \\  "format_version": 1,
        \\  "model_name": "vision-yolo",
        \\  "metadata": {},
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "pipeline.detect", "kind": "Detect", "from": [-1] }
        \\  ],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": []
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var managed = try adapter_factory_mod.initAuto(std.testing.allocator, root_path, .auto);
    defer managed.deinit();

    try std.testing.expect(std.mem.startsWith(u8, managed.descriptorId(), "vision.yolo."));
}

test "adapter factory auto-detects ocr model directories" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "demo.swm", "SWOCR01");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var managed = try adapter_factory_mod.initAuto(std.testing.allocator, root_path, .auto);
    defer managed.deinit();

    try std.testing.expect(std.mem.startsWith(u8, managed.descriptorId(), "ocr.swiftocr."));
}

test "runtime pool acquires and releases owned items" {
    const TestPool = scheduler_mod.RuntimePool(u32);

    const items = try std.testing.allocator.alloc(u32, 2);
    items[0] = 11;
    items[1] = 29;

    var pool = try TestPool.initOwned(std.testing.allocator, items);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 2), pool.len());
    try std.testing.expectEqual(@as(usize, 2), pool.availableCount());

    const lease_a = pool.tryAcquire() orelse return error.ExpectedLease;
    try std.testing.expectEqual(@as(u32, 11), lease_a.item.*);
    try std.testing.expectEqual(@as(usize, 1), pool.availableCount());

    const lease_b = pool.tryAcquire() orelse return error.ExpectedLease;
    try std.testing.expectEqual(@as(u32, 29), lease_b.item.*);
    try std.testing.expectEqual(@as(usize, 0), pool.availableCount());
    try std.testing.expect(pool.tryAcquire() == null);

    lease_a.release();
    try std.testing.expectEqual(@as(usize, 1), pool.availableCount());

    lease_b.release();
    try std.testing.expectEqual(@as(usize, 2), pool.availableCount());
}

test "shared reuse allocator is exposed through engine core" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var reuse = memory.ReuseAllocator.init(gpa.allocator());
    defer reuse.deinit();

    const allocator = reuse.allocator();
    const first = try allocator.alloc(u8, 4096);
    const first_ptr = first.ptr;
    allocator.free(first);

    const second = try allocator.alloc(u8, 4096);
    defer allocator.free(second);

    try std.testing.expectEqual(@intFromPtr(first_ptr), @intFromPtr(second.ptr));
    try std.testing.expectEqual(@as(usize, 1), reuse.snapshot().cache_hits);
}

test "generic plan graph parses component tree and metadata" {
    const raw =
        \\{
        \\  "format_version": 1,
        \\  "model_name": "kinetix-generic",
        \\  "metadata": {
        \\    "modality": "vision",
        \\    "supports_batching": true
        \\  },
        \\  "tensors": [
        \\    { "name": "encoder.weight", "shape": [3, 4], "offset": 0, "nbytes": 48 }
        \\  ],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "pipeline.encoder", "kind": "Encode", "from": [-1] },
        \\    { "index": 1, "path": "pipeline.head", "kind": "Head", "from": [0] }
        \\  ],
        \\  "component_tree": {
        \\    "path": "pipeline",
        \\    "kind": "Pipeline",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "pipeline.encoder",
        \\        "kind": "Encode",
        \\        "attrs": { "hidden": 128 },
        \\        "children": []
        \\      },
        \\      {
        \\        "path": "pipeline.head",
        \\        "kind": "Head",
        \\        "attrs": { "classes": 80 },
        \\        "children": []
        \\      }
        \\    ]
        \\  }
        \\}
    ;

    var parsed = try graph.parseGraph(std.testing.allocator, raw);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 1), parsed.tensors.len);
    try std.testing.expectEqual(@as(usize, 2), parsed.execution_nodes.len);
    try std.testing.expectEqualStrings("kinetix-generic", parsed.model_name);
    try std.testing.expectEqualStrings("vision", parsed.getMetadata("modality").?.asString().?);
    try std.testing.expect(parsed.getMetadata("supports_batching").?.asBool().?);
    try std.testing.expectEqual(@as(usize, 1), parsed.execution_use_counts[0]);
    try std.testing.expectEqual(@as(usize, 0), parsed.execution_use_counts[1]);
    try std.testing.expectEqualStrings("pipeline.head", parsed.execution_components[1].?.path);
    try std.testing.expectEqual(@as(usize, 2), parsed.findTensor("encoder.weight").?.rank());
}

test "generic plan graph preserves compatibility with Axionyx execution path mapping" {
    const raw =
        \\{
        \\  "format_version": 1,
        \\  "model_name": "axionyx-compat",
        \\  "metadata": {},
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "model.0", "kind": "Conv", "from": [-1] }
        \\  ],
        \\  "module_tree": {
        \\    "path": "model",
        \\    "kind": "Root",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "model.model.0",
        \\        "kind": "Conv",
        \\        "attrs": { "stride": [2, 2] },
        \\        "children": []
        \\      }
        \\    ]
        \\  }
        \\}
    ;

    var parsed = try graph.parseGraph(std.testing.allocator, raw);
    defer parsed.deinit();

    try std.testing.expect(parsed.execution_components[0] != null);
    try std.testing.expectEqualStrings("model.model.0", parsed.execution_components[0].?.path);
    try std.testing.expectEqualStrings("Conv", parsed.execution_components[0].?.kind);
}

test "model catalog discovers common artifacts and prefers fastest quantized weights" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.safetensors", "bf16");
    try writeTmpFile(tmp.dir, "model.q6.zinfer", "q6");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try backend.ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    try std.testing.expect(catalog.has(.config));
    try std.testing.expect(catalog.has(.tokenizer_json));
    try std.testing.expect(catalog.has(.safetensors));
    try std.testing.expect(catalog.has(.q6_weights));
    try std.testing.expect(catalog.has(.q8_weights));
    try std.testing.expectEqual(backend.WeightScheme.q8, catalog.resolveAutoScheme());

    const selection = try catalog.resolveWeights(.auto);
    try std.testing.expectEqual(backend.WeightScheme.q8, selection.scheme);
    try std.testing.expect(std.mem.endsWith(u8, selection.path, "model.q8.zinfer"));
}

test "model catalog can discover vision graph and binary weight artifacts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json", "{}");
    try writeTmpFile(tmp.dir, "weights.bin", "weights");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try backend.ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    try std.testing.expect(catalog.has(.graph_json));
    try std.testing.expect(catalog.has(.weights_bin));
    try std.testing.expectEqual(@as(usize, 2), catalog.artifactCount());
    try std.testing.expectError(error.WeightArtifactNotFound, catalog.resolveWeights(.bf16));
}

test "load plan resolves text model artifacts from shared catalog" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.safetensors", "bf16");
    try writeTmpFile(tmp.dir, "model.q4.zinfer", "q4");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try backend.ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    const plan = try load_plan.resolve(&catalog, .{
        .model_dir = root_path,
        .preferred_weights = .auto,
    });

    try std.testing.expect(plan.config_path != null);
    try std.testing.expect(plan.tokenizer_path != null);
    try std.testing.expectEqual(backend.WeightScheme.q4, plan.weight_scheme.?);
    try std.testing.expect(std.mem.endsWith(u8, plan.weights_path.?, "model.q4.zinfer"));
}

test "load plan resolves vision graph and binary weights without tensor backend" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "graph.json", "{}");
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try backend.ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    const plan = try load_plan.resolve(&catalog, .{
        .model_dir = root_path,
    });

    try std.testing.expect(plan.graph_path != null);
    try std.testing.expect(plan.binary_weights_path != null);
    try std.testing.expect(plan.weight_scheme == null);
    try std.testing.expect(plan.weights_path == null);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
