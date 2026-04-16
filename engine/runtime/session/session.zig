const std = @import("std");
const compat = @import("../compat/compat.zig");
const executor_mod = @import("../executor/executor.zig");
const handle_mod = @import("../model/handle.zig");
const planner_mod = @import("../planner/planner.zig");
const types = @import("../types.zig");

pub const OpenModelRequest = struct {
    model_dir: []const u8,
    preferred_weights: types.WeightScheme = .auto,
};

pub const RuntimeSession = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) RuntimeSession {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RuntimeSession) void {
        self.* = undefined;
    }

    pub fn normalizeModel(self: *const RuntimeSession, request: OpenModelRequest) !compat.NormalizedModel {
        return try compat.normalizeModel(self.allocator, request.model_dir, request.preferred_weights);
    }

    pub fn openModel(self: *const RuntimeSession, request: OpenModelRequest) !handle_mod.ModelHandle {
        return .{
            .allocator = self.allocator,
            .normalized = try self.normalizeModel(request),
        };
    }

    pub fn plan(self: *const RuntimeSession, handle: *const handle_mod.ModelHandle, request: types.RuntimeRequest) !types.ExecutionPlan {
        return planner_mod.Planner.init(self.allocator).plan(handle, request);
    }

    pub fn planBatch(
        self: *const RuntimeSession,
        handle: *const handle_mod.ModelHandle,
        request: types.RuntimeBatchRequest,
    ) !types.ExecutionPlan {
        return planner_mod.Planner.init(self.allocator).planBatch(handle, request);
    }

    pub fn execute(self: *const RuntimeSession, handle: *const handle_mod.ModelHandle, execution_plan: *const types.ExecutionPlan) !types.RuntimeResult {
        return executor_mod.Executor.init(self.allocator).execute(handle, execution_plan);
    }

    pub fn executeBatch(self: *const RuntimeSession, handle: *const handle_mod.ModelHandle, execution_plan: *const types.ExecutionPlan) !types.RuntimeBatchResults {
        return executor_mod.Executor.init(self.allocator).executeBatch(handle, execution_plan);
    }
};

test "runtime session opens normalized models through the unified compatibility entrypoint" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    try std.testing.expectEqual(types.ProviderKey.qwen3_text, handle.normalized.provider_key);
    try std.testing.expectEqual(types.Modality.text, handle.normalized.descriptor.modality);
}

test "runtime session can execute qwen3 requests through the unified executor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "generate",
        .input = .{ .text = "hello" },
        .generation = .{
            .max_tokens = 8,
            .native_execution = true,
        },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ExecutionOrigin.native_single_bridge, result.origin);
    try std.testing.expectEqualStrings("text_native_qwen_single", result.note);
    try std.testing.expectEqualStrings("stub-native-single", result.output.text);
}

test "runtime session can execute qwen3 batch requests through the unified executor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"qwen3\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.q8.zinfer", "q8");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    const items = [_]types.RuntimeRequest{
        .{
            .operation = "generate",
            .input = .{ .text = "hello" },
            .generation = .{ .max_tokens = 8, .native_execution = true },
        },
        .{
            .operation = "generate",
            .input = .{ .text = "world" },
            .generation = .{ .max_tokens = 8, .native_execution = true },
        },
    };

    var plan = try session.planBatch(&handle, .{ .items = &items });
    defer plan.deinit();

    var results = try session.executeBatch(&handle, &plan);
    defer results.deinit();

    try std.testing.expectEqual(@as(usize, 2), results.items.len);
    try std.testing.expectEqual(types.ExecutionOrigin.native_batch_bridge, results.items[0].origin);
    try std.testing.expectEqualStrings("text_native_qwen_batch", results.items[0].note);
}

test "runtime session can execute yolo vision requests through the unified executor" {
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
        \\    "children": []
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "detect",
        .input = .{ .image_path = "demo.png" },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ExecutionOrigin.shared_adapter, result.origin);
    try std.testing.expectEqualStrings("vision_shared_detect", result.note);
}

test "runtime session can batch yolo vision requests through the unified executor" {
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
        \\    "children": []
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "weights.bin", "vision");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    const items = [_]types.RuntimeRequest{
        .{
            .operation = "detect",
            .input = .{ .image_path = "demo-a.png" },
        },
        .{
            .operation = "detect",
            .input = .{ .image_path = "demo-b.png" },
        },
    };

    var plan = try session.planBatch(&handle, .{ .items = &items });
    defer plan.deinit();

    var results = try session.executeBatch(&handle, &plan);
    defer results.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.batches.len);
    try std.testing.expectEqual(@as(usize, 2), results.items.len);
    try std.testing.expectEqualStrings("vision_shared_detect", results.items[0].note);
    try std.testing.expectEqualStrings("vision_shared_detect", results.items[1].note);
}

test "runtime session can execute swiftocr requests through the unified executor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeOCRModel(tmp.dir, "demo.swm", 0);
    try writePPMImage(tmp.dir, "demo.ppm", 1, 1, &[_]u8{ 1, 2, 3 });

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.ppm");
    defer std.testing.allocator.free(image_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "infer-ocr",
        .input = .{ .image_path = image_path },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ExecutionOrigin.shared_adapter, result.origin);
    try std.testing.expectEqualStrings("ocr_shared_infer", result.note);
}

test "runtime session can batch swiftocr requests through the unified executor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeOCRModel(tmp.dir, "demo.swm", 0);
    try writePPMImage(tmp.dir, "demo-a.ppm", 1, 1, &[_]u8{ 1, 2, 3 });
    try writePPMImage(tmp.dir, "demo-b.ppm", 1, 1, &[_]u8{ 4, 5, 6 });

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const image_a = try tmp.dir.realpathAlloc(std.testing.allocator, "demo-a.ppm");
    defer std.testing.allocator.free(image_a);
    const image_b = try tmp.dir.realpathAlloc(std.testing.allocator, "demo-b.ppm");
    defer std.testing.allocator.free(image_b);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    const items = [_]types.RuntimeRequest{
        .{
            .operation = "infer-ocr",
            .input = .{ .image_path = image_a },
        },
        .{
            .operation = "infer-ocr",
            .input = .{ .image_path = image_b },
        },
    };

    var plan = try session.planBatch(&handle, .{ .items = &items });
    defer plan.deinit();

    var results = try session.executeBatch(&handle, &plan);
    defer results.deinit();

    try std.testing.expectEqual(@as(usize, 2), plan.batches.len);
    try std.testing.expectEqual(@as(usize, 2), results.items.len);
    try std.testing.expectEqualStrings("ocr_shared_infer", results.items[0].note);
    try std.testing.expectEqualStrings("ocr_shared_infer", results.items[1].note);
}

test "runtime session can normalize and validate chandra document requests" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makeDir("chandra-ocr-2");
    var model_dir = try tmp.dir.openDir("chandra-ocr-2", .{});
    defer model_dir.close();

    try writeTmpFile(model_dir, "config.json", "{\"model_type\":\"qwen3_vl\"}");
    try writeTmpFile(model_dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "demo.pdf", "%PDF-1.7");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, "chandra-ocr-2");
    defer std.testing.allocator.free(root_path);
    const pdf_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.pdf");
    defer std.testing.allocator.free(pdf_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "render-markdown",
        .input = .{ .document_path = pdf_path },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ProviderKey.chandra_ocr, handle.normalized.provider_key);
    try std.testing.expectEqual(types.ExecutionOrigin.shared_adapter, result.origin);
    try std.testing.expectEqual(types.ExecutionNote.validated_only, result.note);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}

fn writeOCRModel(dir: std.fs.Dir, relative_path: []const u8, tensor_count: u32) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();

    var writer_impl = file.writer(&.{});
    const writer = &writer_impl.interface;
    try writer.writeAll(&[_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 });
    try writer.writeInt(u32, tensor_count, .little);
    try writer.flush();
}

fn writePPMImage(dir: std.fs.Dir, relative_path: []const u8, width: usize, height: usize, pixels: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();

    var writer_impl = file.writer(&.{});
    const writer = &writer_impl.interface;
    try writer.print("P6\n{d} {d}\n255\n", .{ width, height });
    try writer.writeAll(pixels);
    try writer.flush();
}
