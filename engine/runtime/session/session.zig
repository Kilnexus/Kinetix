const std = @import("std");
const imaging = @import("Pixio");
const backend_registry = @import("../backend/registry.zig");
const resolver = @import("../model/resolver/resolver.zig");
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

    pub fn normalizeModel(self: *const RuntimeSession, request: OpenModelRequest) !resolver.NormalizedModel {
        return try resolver.normalizeModel(self.allocator, request.model_dir, request.preferred_weights);
    }

    pub fn openModel(self: *const RuntimeSession, request: OpenModelRequest) !handle_mod.ModelHandle {
        var normalized = try self.normalizeModel(request);
        errdefer normalized.deinit();

        const runtime_backend = backend_registry.findByKey(normalized.provider_key) orelse return error.RuntimeExecutionNotImplemented;
        const provider_state = try runtime_backend.open(self.allocator, &normalized);
        errdefer runtime_backend.deinitState(self.allocator, provider_state);

        return .{
            .allocator = self.allocator,
            .normalized = normalized,
            .runtime_backend = runtime_backend,
            .provider_state = provider_state,
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

test "runtime session opens normalized models through the unified resolver entrypoint" {
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
    try std.testing.expect(handle.provider_state != null);
    try std.testing.expectEqual(types.ProviderKey.qwen3_text, handle.runtime_backend.provider_key);
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

    try std.testing.expectEqual(types.ExecutionOrigin.native_single, result.origin);
    try std.testing.expectEqualStrings("text_native_qwen_single", result.note);
    try std.testing.expectEqualStrings("test-native-single", result.output.text);
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
    try std.testing.expectEqual(types.ExecutionOrigin.native_batch, results.items[0].origin);
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

    try std.testing.expectEqual(types.ExecutionOrigin.runtime_backend, result.origin);
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

    try std.testing.expectEqual(types.ExecutionOrigin.runtime_backend, result.origin);
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

test "runtime session reports native chandra readiness without external runtime" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makeDir("chandra-ocr-2");
    var model_dir = try tmp.dir.openDir("chandra-ocr-2", .{});
    defer model_dir.close();

    try writeTmpFile(model_dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 256,
        \\    "vocab_size": 248320,
        \\    "max_position_embeddings": 262144
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 24,
        \\    "hidden_size": 1024,
        \\    "intermediate_size": 4096,
        \\    "num_heads": 16,
        \\    "out_hidden_size": 2560,
        \\    "patch_size": 16,
        \\    "spatial_merge_size": 2,
        \\    "temporal_patch_size": 2,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(model_dir, "tokenizer.json", "{}");
    try writeTmpFile(model_dir, "preprocessor_config.json",
        \\{
        \\  "merge_size": 2,
        \\  "patch_size": 16,
        \\  "temporal_patch_size": 2,
        \\  "size": {
        \\    "longest_edge": 16777216,
        \\    "shortest_edge": 65536
        \\  }
        \\}
    );
    try writeTmpFile(model_dir, "model.safetensors.index.json",
        \\{
        \\  "weight_map": {
        \\    "model.embed_tokens.weight": "model-00001.safetensors",
        \\    "visual.patch_embed.proj.weight": "model-00001.safetensors",
        \\    "visual.merger.mlp.0.weight": "model-00001.safetensors",
        \\    "lm_head.weight": "model-00001.safetensors"
        \\  }
        \\}
    );
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

    try std.testing.expectEqual(types.ExecutionNote.ocr_chandra_native, result.note);
    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"ocr_native_backend_incomplete\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"backend\":\"kinetix_native\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"has_visual_encoder\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"has_multimodal_projector\":true") != null);
}

test "runtime session materializes native chandra markdown output through the unified runtime" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 3,
        \\  "vision_start_token_id": 1,
        \\  "vision_end_token_id": 2,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 8,
        \\    "max_position_embeddings": 1024,
        \\    "rope_parameters": {
        \\      "full_attention": {
        \\        "rope_theta": 250000.0
        \\      },
        \\      "mrope_section": [1, 0, 0]
        \\    }
        \\  },
        \\  "vision_config": {
        \\    "model_type": "qwen3_5",
        \\    "depth": 2,
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_heads": 1,
        \\    "out_hidden_size": 2,
        \\    "patch_size": 2,
        \\    "spatial_merge_size": 1,
        \\    "temporal_patch_size": 1,
        \\    "in_channels": 3
        \\  }
        \\}
    );
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
        \\{
        \\  "do_normalize": false,
        \\  "do_rescale": false,
        \\  "do_resize": false,
        \\  "merge_size": 1,
        \\  "patch_size": 2,
        \\  "temporal_patch_size": 1,
        \\  "size": {
        \\    "longest_edge": 1024,
        \\    "shortest_edge": 1
        \\  }
        \\}
    );
    try writeSyntheticRuntimeReadySafetensors(tmp.dir, "model.safetensors");
    try writeSyntheticTokenizerFiles(tmp.dir);
    try writeSyntheticPng(tmp.dir, "input.png", 4, 2);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "input.png");
    defer std.testing.allocator.free(image_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "render-markdown",
        .input = .{ .image_path = image_path },
        .generation = .{ .max_tokens = 1 },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ExecutionOrigin.native_single, result.origin);
    try std.testing.expectEqual(types.ExecutionNote.ocr_chandra_native, result.note);
    switch (result.output) {
        .text => |value| try std.testing.expectEqualStrings("OCR", value),
        else => return error.ExpectedTextOutput,
    }
}

test "runtime session routes moss tts nano bundles through the unified backend registry" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeMossTtsBundle(tmp.dir);

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    try std.testing.expectEqual(types.ProviderKey.moss_tts_nano_tts, handle.runtime_backend.provider_key);
    try std.testing.expectEqual(types.Modality.tts, handle.normalized.descriptor.modality);

    var plan = try session.plan(&handle, .{
        .operation = "synthesize",
        .input = .{ .text = "hello moss" },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ExecutionOrigin.runtime_backend, result.origin);
    try std.testing.expectEqual(types.ExecutionNote.tts_model_ready, result.note);

    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"provider_key\":\"moss_tts_nano_tts\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"output_contract\":\"audio_path\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"normalized_text\":\"hello moss.\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"chunk_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"sample_rate\":48000") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"builtin_voice_count\":2") != null);
}

test "runtime session routes bert models through the unified backend registry" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json", "{\"model_type\":\"bert\"}");
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "model.safetensors", "bert");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "fill-mask",
        .input = .{ .text = "kinetix [MASK]" },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ProviderKey.bert_text, handle.runtime_backend.provider_key);
    try std.testing.expectEqual(types.ExecutionNote.validated_only, result.note);

    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"provider_key\":\"bert_text\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"runtime_backend_ready\"") != null);
}

test "runtime session routes generic models through the unified backend registry" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "blob.bin", "generic");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var session = RuntimeSession.init(std.testing.allocator);
    defer session.deinit();

    var handle = try session.openModel(.{ .model_dir = root_path });
    defer handle.deinit();

    var plan = try session.plan(&handle, .{
        .operation = "infer",
        .input = .{ .text = "probe" },
    });
    defer plan.deinit();

    var result = try session.execute(&handle, &plan);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(types.ProviderKey.generic, handle.runtime_backend.provider_key);
    try std.testing.expectEqual(types.ExecutionNote.validated_only, result.note);

    const payload = switch (result.output) {
        .json => |value| value,
        else => return error.ExpectedJsonOutput,
    };
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"provider_key\":\"generic\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"runtime_backend_ready\"") != null);
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

const SyntheticTensorSpec = struct {
    name: []const u8,
    shape: []const usize,
    values: []const f32,
};

fn writeSyntheticPng(dir: std.fs.Dir, relative_path: []const u8, width: usize, height: usize) !void {
    var image = try imaging.ImageU8.init(std.testing.allocator, width, height, 3);
    defer image.deinit();

    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }

    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);

    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(encoded);
}

fn writeSyntheticTokenizerFiles(dir: std.fs.Dir) !void {
    try writeTmpFile(dir, "vocab.json", "{}");
    try writeTmpFile(dir, "merges.txt", "# synthetic\n");
    try writeTmpFile(dir, "tokenizer_config.json",
        \\{
        \\  "added_tokens_decoder": {
        \\    "4": { "content": "OCR" },
        \\    "7": {
        \\      "content": "<|im_start|>user\nRead the document image and transcribe it as markdown.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        \\    }
        \\  }
        \\}
    );
}

fn writeSyntheticRuntimeReadySafetensors(dir: std.fs.Dir, relative_path: []const u8) !void {
    const patch_weight_shape = [_]usize{ 2, 3, 1, 2, 2 };
    const patch_bias_shape = [_]usize{2};
    const pos_shape = [_]usize{ 4, 2 };
    const qkv_shape = [_]usize{ 6, 2 };
    const linear_2x2_shape = [_]usize{ 2, 2 };
    const mlp_fc1_shape = [_]usize{ 3, 2 };
    const mlp_fc2_shape = [_]usize{ 2, 3 };
    const embed_shape = [_]usize{ 8, 2 };
    const norm_shape = [_]usize{2};
    const gate_shape = [_]usize{ 8, 2 };
    const down_shape = [_]usize{ 2, 8 };

    const patch_weights = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    const patch_bias = [_]f32{ 0, 1 };
    const visual_pos = [_]f32{
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    };
    const block_norm_weight = [_]f32{ 1, 1 };
    const block_norm_bias = [_]f32{ 0, 0 };
    const block_qkv = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    const block_qkv_bias = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const block_proj = [_]f32{
        1, 0,
        0, 1,
    };
    const block_proj_bias = [_]f32{ 0, 0 };
    const block_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const block_fc1_bias = [_]f32{ 0, 0, 0.5 };
    const block_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const block_fc2_bias = [_]f32{ 0, 0 };
    const merger_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    const merger_fc1_bias = [_]f32{ 0, 0, 0.5 };
    const merger_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    const merger_fc2_bias = [_]f32{ 0.25, -0.25 };
    const embed_tokens = [_]f32{
        0,   0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0.5, 0.5,
        0.6, 0.6,
        0.7, 0.7,
    };
    const final_norm = [_]f32{ 1, 1 };
    const q_norm = [_]f32{ 1, 1 };
    const zero_2x2 = [_]f32{0} ** 4;
    const zero_gate = [_]f32{0} ** 16;
    const zero_down = [_]f32{0} ** 16;
    const lm_head = [_]f32{
        0,  0,
        0,  0,
        0,  0,
        0,  0,
        10, 10,
        0,  0,
        0,  0,
        0,  0,
    };

    const specs = [_]SyntheticTensorSpec{
        .{ .name = "visual.patch_embed.proj.weight", .shape = &patch_weight_shape, .values = &patch_weights },
        .{ .name = "visual.patch_embed.proj.bias", .shape = &patch_bias_shape, .values = &patch_bias },
        .{ .name = "visual.pos_embed", .shape = &pos_shape, .values = &visual_pos },
        .{ .name = "visual.blocks.0.norm1.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.0.norm1.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.0.attn.qkv.weight", .shape = &qkv_shape, .values = &block_qkv },
        .{ .name = "visual.blocks.0.attn.qkv.bias", .shape = &[_]usize{6}, .values = &block_qkv_bias },
        .{ .name = "visual.blocks.0.attn.proj.weight", .shape = &linear_2x2_shape, .values = &block_proj },
        .{ .name = "visual.blocks.0.attn.proj.bias", .shape = &norm_shape, .values = &block_proj_bias },
        .{ .name = "visual.blocks.0.norm2.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.0.norm2.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.0.mlp.linear_fc1.weight", .shape = &mlp_fc1_shape, .values = &block_fc1 },
        .{ .name = "visual.blocks.0.mlp.linear_fc1.bias", .shape = &[_]usize{3}, .values = &block_fc1_bias },
        .{ .name = "visual.blocks.0.mlp.linear_fc2.weight", .shape = &mlp_fc2_shape, .values = &block_fc2 },
        .{ .name = "visual.blocks.0.mlp.linear_fc2.bias", .shape = &norm_shape, .values = &block_fc2_bias },
        .{ .name = "visual.blocks.1.norm1.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.1.norm1.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.1.attn.qkv.weight", .shape = &qkv_shape, .values = &block_qkv },
        .{ .name = "visual.blocks.1.attn.qkv.bias", .shape = &[_]usize{6}, .values = &block_qkv_bias },
        .{ .name = "visual.blocks.1.attn.proj.weight", .shape = &linear_2x2_shape, .values = &block_proj },
        .{ .name = "visual.blocks.1.attn.proj.bias", .shape = &norm_shape, .values = &block_proj_bias },
        .{ .name = "visual.blocks.1.norm2.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "visual.blocks.1.norm2.bias", .shape = &norm_shape, .values = &block_norm_bias },
        .{ .name = "visual.blocks.1.mlp.linear_fc1.weight", .shape = &mlp_fc1_shape, .values = &block_fc1 },
        .{ .name = "visual.blocks.1.mlp.linear_fc1.bias", .shape = &[_]usize{3}, .values = &block_fc1_bias },
        .{ .name = "visual.blocks.1.mlp.linear_fc2.weight", .shape = &mlp_fc2_shape, .values = &block_fc2 },
        .{ .name = "visual.blocks.1.mlp.linear_fc2.bias", .shape = &norm_shape, .values = &block_fc2_bias },
        .{ .name = "visual.merger.mlp.0.weight", .shape = &mlp_fc1_shape, .values = &merger_fc1 },
        .{ .name = "visual.merger.mlp.0.bias", .shape = &[_]usize{3}, .values = &merger_fc1_bias },
        .{ .name = "visual.merger.mlp.2.weight", .shape = &mlp_fc2_shape, .values = &merger_fc2 },
        .{ .name = "visual.merger.mlp.2.bias", .shape = &norm_shape, .values = &merger_fc2_bias },
        .{ .name = "model.embed_tokens.weight", .shape = &embed_shape, .values = &embed_tokens },
        .{ .name = "model.norm.weight", .shape = &norm_shape, .values = &final_norm },
        .{ .name = "lm_head.weight", .shape = &embed_shape, .values = &lm_head },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "model.layers.0.self_attn.q_norm.weight", .shape = &norm_shape, .values = &q_norm },
        .{ .name = "model.layers.0.self_attn.k_norm.weight", .shape = &norm_shape, .values = &q_norm },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &linear_2x2_shape, .values = &zero_2x2 },
        .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &norm_shape, .values = &block_norm_weight },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &gate_shape, .values = &zero_gate },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &gate_shape, .values = &zero_gate },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &down_shape, .values = &zero_down },
    };

    try writeSyntheticF32Safetensors(std.testing.allocator, dir, relative_path, &specs);
}

fn writeSyntheticF32Safetensors(
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
    relative_path: []const u8,
    specs: []const SyntheticTensorSpec,
) !void {
    var header = std.ArrayList(u8).init(allocator);
    defer header.deinit();

    var offsets = try allocator.alloc(u64, specs.len + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (specs, 0..) |spec, index| {
        const tensor_elements = try tensorElementCount(spec.shape);
        if (tensor_elements != spec.values.len) return error.ShapeMismatch;
        offsets[index + 1] = offsets[index] + spec.values.len * @sizeOf(f32);
    }

    try header.append('{');
    for (specs, 0..) |spec, index| {
        if (index != 0) try header.append(',');
        try header.writer().print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{spec.name});
        for (spec.shape, 0..) |dim, dim_index| {
            if (dim_index != 0) try header.append(',');
            try header.writer().print("{d}", .{dim});
        }
        try header.writer().print("],\"data_offsets\":[{d},{d}]}}", .{
            offsets[index],
            offsets[index + 1],
        });
    }
    try header.append('}');

    var file = try dir.createFile(relative_path, .{});
    defer file.close();

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.items.len, .little);
    try file.writeAll(&length_prefix);
    try file.writeAll(header.items);

    for (specs) |spec| {
        for (spec.values) |value| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
            try file.writeAll(&bytes);
        }
    }
}

fn tensorElementCount(shape: []const usize) !usize {
    var total: usize = 1;
    for (shape) |dim| {
        total = try std.math.mul(usize, total, dim);
    }
    return total;
}
