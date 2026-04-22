const std = @import("std");
const imaging = @import("Pixio");
const core = @import("../model/core.zig");
const input = @import("../input/loader.zig");
const exec = @import("../execute/runner.zig");
const decoder_types = @import("../../../../../text/decoder_types.zig");

const io = core.io;

test "native chandra config parser accepts qwen3_5 document vl config" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "architectures": ["Qwen3_5ForConditionalGeneration"],
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "video_token_id": 248057,
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

    const config_path = try tmp.dir.realPathFileAlloc(io, "config.json", std.testing.allocator);
    defer std.testing.allocator.free(config_path);

    var parsed = try core.loadConfigFromFile(std.testing.allocator, config_path);
    defer parsed.deinit();

    try std.testing.expect(parsed.value.isSupportedChandraShape());
    try std.testing.expectEqual(@as(usize, 248056), parsed.value.image_token_id);
    try std.testing.expectEqual(@as(usize, 24), parsed.value.vision_config.depth);
    try std.testing.expectEqual(@as(usize, 2560), parsed.value.text_config.hidden_size);
}

test "native chandra inspect reports tensor manifest readiness" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
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
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
    try writeTmpFile(tmp.dir, "preprocessor_config.json",
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
    try writeTmpFile(tmp.dir, "model.safetensors.index.json",
        \\{
        \\  "weight_map": {
        \\    "model.embed_tokens.weight": "model-00001.safetensors",
        \\    "visual.patch_embed.proj.weight": "model-00001.safetensors",
        \\    "visual.merger.mlp.0.weight": "model-00001.safetensors",
        \\    "lm_head.weight": "model-00001.safetensors"
        \\  }
        \\}
    );

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);

    const readiness = core.inspect(root_path);
    try std.testing.expect(readiness.has_config);
    try std.testing.expect(readiness.has_supported_config);
    try std.testing.expect(readiness.has_weights);
    try std.testing.expect(readiness.has_visual_encoder);
    try std.testing.expect(readiness.has_patch_embedding_weight);
    try std.testing.expect(readiness.has_multimodal_projector);
    try std.testing.expect(readiness.has_image_processor_config);
    try std.testing.expect(readiness.has_document_preprocessor);
    try std.testing.expectEqual(@as(usize, 1), readiness.text_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.vision_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.projector_tensor_count);
    try std.testing.expectEqual(@as(usize, 1), readiness.output_tensor_count);
}

test "native chandra execute preprocesses image and runs patch embedding stage" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "config.json",
        \\{
        \\  "model_type": "qwen3_5",
        \\  "image_token_id": 248056,
        \\  "vision_start_token_id": 248053,
        \\  "vision_end_token_id": 248054,
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2,
        \\    "intermediate_size": 8,
        \\    "num_hidden_layers": 1,
        \\    "num_attention_heads": 1,
        \\    "num_key_value_heads": 1,
        \\    "head_dim": 2,
        \\    "vocab_size": 32,
        \\    "max_position_embeddings": 1024
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
    try writeTmpFile(tmp.dir, "tokenizer.json", "{}");
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
    try writeSyntheticPatchEmbeddingSafetensors(tmp.dir, "model.safetensors");

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try exec.execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"visual_merger\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_embedding_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_embedding_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_position_embedding_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_position_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"vision_block_depth\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_block_attention_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_attention_blocks_executed\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_attention_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_block_mlp_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_mlp_blocks_executed\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_merger_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"visual_token_dim\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"patch_token_count\":2") != null);
}

test "chandra input loader prepares frame sequence from directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeSolidPng(tmp.dir, "frame_02.png", 2, 2, 255);
    try writeSolidPng(tmp.dir, "frame_01.png", 2, 2, 0);

    const frames_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(frames_path);

    var prepared = try input.loadPreparedInputFromDirectory(std.testing.allocator, frames_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 2), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(usize, 1), prepared.grid.temporal_patch_count);
    try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[0]);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
}

test "chandra input loader prepares frame sequence from manifest" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeSolidPng(tmp.dir, "a.png", 2, 2, 64);
    try writeSolidPng(tmp.dir, "b.png", 2, 2, 192);
    try writeTmpFile(tmp.dir, "frames.frames",
        \\# ordered frame list
        \\b.png
        \\a.png
    );

    const manifest_path = try tmp.dir.realPathFileAlloc(io, "frames.frames", std.testing.allocator);
    defer std.testing.allocator.free(manifest_path);

    var prepared = try input.loadPreparedInputFromManifest(std.testing.allocator, manifest_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    try std.testing.expectEqual(@as(usize, 2), prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 192.0), prepared.tensor.data[0]);
    try std.testing.expectEqual(@as(f32, 64.0), prepared.tensor.data[prepared.tensor.stride_n]);
}

test "chandra input loader handles gif via pixio codec" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedGif(tmp.dir, "animated.gif");
    const gif_path = try tmp.dir.realPathFileAlloc(io, "animated.gif", std.testing.allocator);
    defer std.testing.allocator.free(gif_path);

    var prepared = try input.loadPreparedInputFromPath(std.testing.allocator, gif_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) 2 else 1;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 2) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    } else {
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    }
}

test "chandra input loader handles webp via pixio codec" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedWebp(tmp.dir, "animated.webp");
    const webp_path = try tmp.dir.realPathFileAlloc(io, "animated.webp", std.testing.allocator);
    defer std.testing.allocator.free(webp_path);

    var prepared = try input.loadPreparedInputFromPath(std.testing.allocator, webp_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 2,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) 2 else 1;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(usize, 2), prepared.grid.frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 2) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    } else {
        try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n + 1]);
    }
}

test "chandra directory loader expands animated webp entries into frame sequence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedWebp(tmp.dir, "000.webp");
    try writeSolidPng(tmp.dir, "001.png", 1, 1, 32);
    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir_path);

    var prepared = try input.loadPreparedInputFromPath(std.testing.allocator, dir_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 4,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) 3 else 2;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 3) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 32.0), prepared.tensor.data[prepared.tensor.stride_n * 2]);
    } else {
        try std.testing.expectEqual(@as(f32, 32.0), prepared.tensor.data[prepared.tensor.stride_n]);
    }
}

test "chandra manifest loader expands animated gif entries into frame sequence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeAnimatedGif(tmp.dir, "clip.gif");
    try writeSolidPng(tmp.dir, "tail.png", 1, 1, 48);
    try writeTmpFile(tmp.dir, "frames.lst",
        \\clip.gif
        \\tail.png
        \\
    );
    const manifest_path = try tmp.dir.realPathFileAlloc(io, "frames.lst", std.testing.allocator);
    defer std.testing.allocator.free(manifest_path);

    var prepared = try input.loadPreparedInputFromPath(std.testing.allocator, manifest_path, .{
        .do_normalize = false,
        .do_rescale = false,
        .do_resize = false,
        .merge_size = 1,
        .patch_size = 1,
        .temporal_patch_size = 4,
        .size = .{
            .longest_edge = 1024,
            .shortest_edge = 1,
        },
    });
    defer prepared.deinit();

    const expected_frame_count: usize = if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) 3 else 2;
    try std.testing.expectEqual(expected_frame_count, prepared.grid.source_frame_count);
    try std.testing.expectEqual(@as(f32, 255.0), prepared.tensor.data[0]);
    if (expected_frame_count == 3) {
        try std.testing.expectEqual(@as(f32, 0.0), prepared.tensor.data[prepared.tensor.stride_n]);
        try std.testing.expectEqual(@as(f32, 48.0), prepared.tensor.data[prepared.tensor.stride_n * 2]);
    } else {
        try std.testing.expectEqual(@as(f32, 48.0), prepared.tensor.data[prepared.tensor.stride_n]);
    }
}

test "multimodal mrope plan continues text after visual axis max" {
    const position_plan = core.MultimodalPositionPlan.init(.mrope, 6, 1, 3, 2, 2);

    try std.testing.expectEqual(@as(usize, 0), position_plan.vision_start_position);
    try std.testing.expectEqual(@as(usize, 1), position_plan.visual_start_position);
    try std.testing.expectEqual(@as(usize, 4), position_plan.vision_end_position);
    try std.testing.expectEqual(@as(usize, 5), position_plan.prompt_start_position);
    try std.testing.expectEqual(@as(usize, 7), position_plan.generation_start_position);
    try std.testing.expectEqual(@as(usize, 10), position_plan.total_prefill_tokens);

    const first_visual = core.visualTokenPosition(.mrope, position_plan.visual_start_position, 0, 1, 2, 3);
    try std.testing.expectEqual(decoder_types.RopePositionMode.mrope, first_visual.mode);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, &first_visual.axes);

    const tail_visual = core.visualTokenPosition(.mrope, position_plan.visual_start_position, 5, 1, 2, 3);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 3 }, &tail_visual.axes);

    const prompt_position = core.textTokenPosition(.mrope, position_plan.prompt_start_position);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5, 5 }, &prompt_position.axes);
}

test "visual token position maps thw axes for future multi-frame layout" {
    const position = core.visualTokenPosition(.mrope, 4, 7, 2, 2, 2);
    try std.testing.expectEqual(decoder_types.RopePositionMode.mrope, position.mode);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5, 5 }, &position.axes);
}

test "native chandra execute exposes mrope prefill metadata" {
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

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try exec.execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
        .max_output_tokens = 0,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"text_prefill\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"text_prefill_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoder_rope_position_mode\":\"mrope\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoder_mrope_sections\":[1,0,0,0]") != null);
}

test "native chandra execute decodes content with synthetic tokenizer" {
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

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const payload = try exec.execute(std.testing.allocator, .{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
        .max_output_tokens = 1,
    });
    defer std.testing.allocator.free(payload);

    try std.testing.expect(std.mem.indexOf(u8, payload, "\"native_stage\":\"text_decode\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"ocr_native_text_decoded_partial\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"text_decode_executed\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"decoded_token_count\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"content\":\"OCR\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"markdown\":\"OCR\"") != null);
}

test "native chandra detailed execution materializes markdown output" {
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

    var image = try imaging.ImageU8.init(std.testing.allocator, 4, 2, 3);
    defer image.deinit();
    for (0..image.width * image.height) |pixel_index| {
        const value: u8 = @intCast(pixel_index + 1);
        image.data[pixel_index * 3] = value;
        image.data[pixel_index * 3 + 1] = 0;
        image.data[pixel_index * 3 + 2] = 0;
    }
    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try tmp.dir.writeFile(io, .{ .sub_path = "input.png", .data = encoded });

    const root_path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root_path);
    const image_path = try tmp.dir.realPathFileAlloc(io, "input.png", std.testing.allocator);
    defer std.testing.allocator.free(image_path);

    const context = core.Context{
        .operation = "render-markdown",
        .model_path = root_path,
        .input_path = image_path,
        .execution = .sync,
        .max_output_tokens = 1,
    };

    var result = try exec.executeDetailed(std.testing.allocator, context);
    defer result.deinit(std.testing.allocator);

    var materialized = (try result.materializeOutput(std.testing.allocator, context.operation)) orelse return error.ExpectedMaterializedOutput;
    defer materialized.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("OCR", materialized.text);

    const payload = try result.toJsonAlloc(std.testing.allocator, context);
    defer std.testing.allocator.free(payload);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"status\":\"ocr_native_text_decoded_partial\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, payload, "\"markdown\":\"OCR\"") != null);
}

fn writeTmpFile(dir: std.Io.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);

    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;
    try writer.writeAll(contents);
    try writer.flush();
}

fn writeSolidPng(dir: std.Io.Dir, relative_path: []const u8, width: usize, height: usize, value: u8) !void {
    var image = try imaging.ImageU8.init(std.testing.allocator, width, height, 3);
    defer image.deinit();
    image.fill(value);

    const encoded = try imaging.encodePngAlloc(std.testing.allocator, &image);
    defer std.testing.allocator.free(encoded);
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = encoded });
}

fn writeAnimatedGif(dir: std.Io.Dir, relative_path: []const u8) !void {
    const gif_bytes = [_]u8{
        'G',  'I',  'F',  '8',  '9',  'a',
        0x01, 0x00, 0x01, 0x00, 0x80, 0x00,
        0x00, 0xff, 0x00, 0x00, 0x00, 0xff,
        0x00, 0x21, 0xf9, 0x04, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x00, 0x02, 0x02, 0x44, 0x01, 0x00,
        0x21, 0xf9, 0x04, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x2c, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0x02, 0x02, 0x4c, 0x01, 0x00, 0x3b,
    };
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = &gif_bytes });
}

fn writeAnimatedWebp(dir: std.Io.Dir, relative_path: []const u8) !void {
    const webp_bytes = [_]u8{
        0x52, 0x49, 0x46, 0x46, 0x84, 0x00, 0x00, 0x00,
        0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38, 0x58,
        0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x4e,
        0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46,
        0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x64, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c,
        0x0f, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
        0x00, 0x07, 0x10, 0xfd, 0x8f, 0xfe, 0x07, 0x22,
        0xa2, 0xff, 0x01, 0x00, 0x41, 0x4e, 0x4d, 0x46,
        0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc8, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c,
        0x0f, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
        0x00, 0x07, 0xd0, 0xff, 0x88, 0xfe, 0x07, 0x22,
        0xa2, 0xff, 0x01, 0x00,
    };
    try dir.writeFile(io, .{ .sub_path = relative_path, .data = &webp_bytes });
}

fn writeSyntheticPatchEmbeddingSafetensors(dir: std.Io.Dir, relative_path: []const u8) !void {
    const header =
        \\{"visual.patch_embed.proj.weight":{"dtype":"F32","shape":[2,3,1,2,2],"data_offsets":[0,96]},"visual.patch_embed.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[96,104]},"visual.pos_embed":{"dtype":"F32","shape":[4,2],"data_offsets":[104,136]},"visual.blocks.0.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[136,144]},"visual.blocks.0.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[144,152]},"visual.blocks.0.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[152,200]},"visual.blocks.0.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[200,224]},"visual.blocks.0.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[224,240]},"visual.blocks.0.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[240,248]},"visual.blocks.0.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[248,256]},"visual.blocks.0.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[256,264]},"visual.blocks.0.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[264,288]},"visual.blocks.0.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[288,300]},"visual.blocks.0.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[300,324]},"visual.blocks.0.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[324,332]},"visual.blocks.1.norm1.weight":{"dtype":"F32","shape":[2],"data_offsets":[332,340]},"visual.blocks.1.norm1.bias":{"dtype":"F32","shape":[2],"data_offsets":[340,348]},"visual.blocks.1.attn.qkv.weight":{"dtype":"F32","shape":[6,2],"data_offsets":[348,396]},"visual.blocks.1.attn.qkv.bias":{"dtype":"F32","shape":[6],"data_offsets":[396,420]},"visual.blocks.1.attn.proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[420,436]},"visual.blocks.1.attn.proj.bias":{"dtype":"F32","shape":[2],"data_offsets":[436,444]},"visual.blocks.1.norm2.weight":{"dtype":"F32","shape":[2],"data_offsets":[444,452]},"visual.blocks.1.norm2.bias":{"dtype":"F32","shape":[2],"data_offsets":[452,460]},"visual.blocks.1.mlp.linear_fc1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[460,484]},"visual.blocks.1.mlp.linear_fc1.bias":{"dtype":"F32","shape":[3],"data_offsets":[484,496]},"visual.blocks.1.mlp.linear_fc2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[496,520]},"visual.blocks.1.mlp.linear_fc2.bias":{"dtype":"F32","shape":[2],"data_offsets":[520,528]},"visual.merger.mlp.0.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[528,552]},"visual.merger.mlp.0.bias":{"dtype":"F32","shape":[3],"data_offsets":[552,564]},"visual.merger.mlp.2.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[564,588]},"visual.merger.mlp.2.bias":{"dtype":"F32","shape":[2],"data_offsets":[588,596]},"model.embed_tokens.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[596,604]},"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[604,612]}}
    ;

    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);
    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.len, .little);
    try writer.writeAll(&length_prefix);
    try writer.writeAll(header);

    var payload: [612]u8 = undefined;
    @memset(&payload, 0);

    const patch_weights = [_]f32{
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        2, 2, 2, 2,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    writeF32Slice(&payload, 0, &patch_weights);
    writeF32Scalar(&payload, 96, 0.0);
    writeF32Scalar(&payload, 100, 1.0);

    const visual_pos = [_]f32{
        1, 10,
        2, 20,
        3, 30,
        4, 40,
    };
    writeF32Slice(&payload, 104, &visual_pos);

    const block_qkv = [_]f32{
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };
    const block_proj = [_]f32{
        1, 0,
        0, 1,
    };
    const block_norm_weight = [_]f32{ 1, 1 };
    const block_norm_bias = [_]f32{ 0, 0 };
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
    const block_proj_bias = [_]f32{ 0, 0 };
    const block_qkv_bias = [_]f32{ 0, 0, 0, 0, 0, 0 };
    const block_fc2_bias = [_]f32{ 0, 0 };

    writeSyntheticVisionBlock(&payload, 136, block_norm_weight[0..], block_norm_bias[0..], block_qkv[0..], block_qkv_bias[0..], block_proj[0..], block_proj_bias[0..], block_norm_weight[0..], block_norm_bias[0..], block_fc1[0..], block_fc1_bias[0..], block_fc2[0..], block_fc2_bias[0..]);
    writeSyntheticVisionBlock(&payload, 332, block_norm_weight[0..], block_norm_bias[0..], block_qkv[0..], block_qkv_bias[0..], block_proj[0..], block_proj_bias[0..], block_norm_weight[0..], block_norm_bias[0..], block_fc1[0..], block_fc1_bias[0..], block_fc2[0..], block_fc2_bias[0..]);

    const merger_fc1 = [_]f32{
        1, 0,
        0, 1,
        1, 1,
    };
    writeF32Slice(&payload, 528, &merger_fc1);
    writeF32Scalar(&payload, 552, 0.0);
    writeF32Scalar(&payload, 556, 0.0);
    writeF32Scalar(&payload, 560, 0.5);

    const merger_fc2 = [_]f32{
        1, 0, 0,
        0, 1, 1,
    };
    writeF32Slice(&payload, 564, &merger_fc2);
    writeF32Scalar(&payload, 588, 0.25);
    writeF32Scalar(&payload, 592, -0.25);

    try writer.writeAll(&payload);
    try writer.flush();
}

const SyntheticTensorSpec = struct {
    name: []const u8,
    shape: []const usize,
    values: []const f32,
};

fn writeSyntheticRuntimeReadySafetensors(dir: std.Io.Dir, relative_path: []const u8) !void {
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

fn writeSyntheticTokenizerFiles(dir: std.Io.Dir) !void {
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

fn writeSyntheticF32Safetensors(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
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

    var file = try dir.createFile(io, relative_path, .{});
    defer file.close(io);
    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;

    var length_prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &length_prefix, header.items.len, .little);
    try writer.writeAll(&length_prefix);
    try writer.writeAll(header.items);

    for (specs) |spec| {
        for (spec.values) |value| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
            try writer.writeAll(&bytes);
        }
    }
    try writer.flush();
}

fn tensorElementCount(shape: []const usize) !usize {
    var total: usize = 1;
    for (shape) |dim| {
        total = try std.math.mul(usize, total, dim);
    }
    return total;
}

fn writeSyntheticVisionBlock(
    payload: []u8,
    offset: usize,
    norm1_weight: []const f32,
    norm1_bias: []const f32,
    qkv_weight: []const f32,
    qkv_bias: []const f32,
    proj_weight: []const f32,
    proj_bias: []const f32,
    norm2_weight: []const f32,
    norm2_bias: []const f32,
    fc1_weight: []const f32,
    fc1_bias: []const f32,
    fc2_weight: []const f32,
    fc2_bias: []const f32,
) void {
    writeF32Slice(payload, offset, norm1_weight);
    writeF32Slice(payload, offset + 8, norm1_bias);
    writeF32Slice(payload, offset + 16, qkv_weight);
    writeF32Slice(payload, offset + 64, qkv_bias);
    writeF32Slice(payload, offset + 88, proj_weight);
    writeF32Slice(payload, offset + 104, proj_bias);
    writeF32Slice(payload, offset + 112, norm2_weight);
    writeF32Slice(payload, offset + 120, norm2_bias);
    writeF32Slice(payload, offset + 128, fc1_weight);
    writeF32Slice(payload, offset + 152, fc1_bias);
    writeF32Slice(payload, offset + 164, fc2_weight);
    writeF32Slice(payload, offset + 188, fc2_bias);
}

fn writeF32Slice(payload: []u8, offset: usize, values: []const f32) void {
    for (values, 0..) |value, index| {
        writeF32Scalar(payload, offset + index * 4, value);
    }
}

fn writeF32Scalar(payload: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, payload[offset .. offset + 4][0..4], @bitCast(value), .little);
}
