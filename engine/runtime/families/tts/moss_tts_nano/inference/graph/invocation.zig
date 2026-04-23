const std = @import("std");
const shared_graph = @import("../../../../../shared/graph/index.zig");
const bundle = @import("../../bundle/index.zig");
const request = @import("../request.zig");

pub const bindings = shared_graph.bindings;

pub fn buildPrefillInvocation(
    allocator: std.mem.Allocator,
    tts: bundle.TtsSummary,
    rows: request.RequestRows,
) !bindings.GraphInvocation {
    if (tts.files.prefill.len == 0) return error.MissingPrefillGraph;
    if (tts.onnx.prefill_output_names.len == 0) return error.MissingPrefillOutputs;
    if (rows.input_ids.len == 0 or rows.row_width == 0) return error.InvalidPrefillRows;

    const inputs = try allocator.alloc(bindings.TensorBinding, 2);
    var filled: usize = 0;
    errdefer {
        for (inputs[0..filled]) |*input| input.deinit();
        allocator.free(inputs);
    }

    var input_ids_values = try flattenRows(allocator, rows.input_ids, rows.row_width);
    errdefer allocator.free(input_ids_values);
    inputs[0] = try bindings.TensorBinding.init(
        allocator,
        "input_ids",
        .i32,
        &.{ 1, rows.sequenceLength(), rows.row_width },
        .{ .i32 = input_ids_values },
    );
    input_ids_values = &.{};
    filled += 1;

    const attention_width = if (rows.attention_mask.len == 0) 0 else rows.attention_mask[0].len;
    var attention_values = try flattenRows(allocator, rows.attention_mask, attention_width);
    errdefer allocator.free(attention_values);
    inputs[1] = try bindings.TensorBinding.init(
        allocator,
        "attention_mask",
        .i32,
        &.{ 1, rows.sequenceLength() },
        .{ .i32 = attention_values },
    );
    attention_values = &.{};
    filled += 1;

    return try bindings.GraphInvocation.init(
        allocator,
        "moss_tts_prefill",
        tts.files.prefill,
        inputs,
        tts.onnx.prefill_output_names,
    );
}

pub fn buildDecodeStepTemplate(
    allocator: std.mem.Allocator,
    tts: bundle.TtsSummary,
    row_width: usize,
    cache_input_count: usize,
) !bindings.GraphInvocation {
    if (tts.files.decode_step.len == 0) return error.MissingDecodeGraph;
    if (tts.onnx.decode_output_names.len == 0) return error.MissingDecodeOutputs;

    const base_inputs: usize = 2;
    const input_count = base_inputs + cache_input_count;
    const inputs = try allocator.alloc(bindings.TensorBinding, input_count);
    var filled: usize = 0;
    errdefer {
        for (inputs[0..filled]) |*input| input.deinit();
        allocator.free(inputs);
    }

    inputs[0] = try bindings.TensorBinding.init(allocator, "input_ids", .i32, &.{ 1, 1, row_width }, .none);
    filled += 1;
    inputs[1] = try bindings.TensorBinding.init(allocator, "past_valid_lengths", .i32, &.{1}, .none);
    filled += 1;

    var index: usize = 0;
    while (index < cache_input_count) : (index += 1) {
        const name = if (index + base_inputs < tts.onnx.decode_input_names.len)
            tts.onnx.decode_input_names[index + base_inputs]
        else
            "past";
        inputs[filled] = try bindings.TensorBinding.init(allocator, name, .f32, &.{}, .none);
        filled += 1;
    }

    return try bindings.GraphInvocation.init(
        allocator,
        "moss_tts_decode_step",
        tts.files.decode_step,
        inputs,
        tts.onnx.decode_output_names,
    );
}

pub fn buildCodecDecodeTemplate(
    allocator: std.mem.Allocator,
    codec: bundle.CodecConfig,
    frame_count: usize,
) !bindings.GraphInvocation {
    if (codec.files.decode_full.len == 0) return error.MissingCodecDecodeGraph;
    if (codec.onnx.decode_output_names.len == 0) return error.MissingCodecDecodeOutputs;

    const inputs = try allocator.alloc(bindings.TensorBinding, 2);
    var filled: usize = 0;
    errdefer {
        for (inputs[0..filled]) |*input| input.deinit();
        allocator.free(inputs);
    }

    inputs[0] = try bindings.TensorBinding.init(
        allocator,
        "audio_codes",
        .i32,
        &.{ 1, frame_count, codec.num_quantizers },
        .none,
    );
    filled += 1;
    inputs[1] = try bindings.TensorBinding.init(allocator, "audio_code_lengths", .i32, &.{1}, .none);
    filled += 1;

    return try bindings.GraphInvocation.init(
        allocator,
        "moss_codec_decode_full",
        codec.files.decode_full,
        inputs,
        codec.onnx.decode_output_names,
    );
}

fn flattenRows(allocator: std.mem.Allocator, rows: []const []const i32, row_width: usize) ![]i32 {
    if (row_width == 0) return error.InvalidTensorShape;
    const out = try allocator.alloc(i32, rows.len * row_width);
    errdefer allocator.free(out);
    var offset: usize = 0;
    for (rows) |row| {
        if (row.len != row_width) return error.InvalidTensorShape;
        @memcpy(out[offset .. offset + row_width], row);
        offset += row_width;
    }
    return out;
}

test "moss graph invocation builds concrete prefill feed" {
    const templates = @import("../../bundle/manifest.zig").PromptTemplates{
        .user_prompt_prefix_token_ids = &.{10},
        .user_prompt_after_reference_token_ids = &.{11},
        .assistant_prompt_prefix_token_ids = &.{12},
    };
    const prompt_audio_codes = [_][]const i32{&.{ 1, 2 }};
    var rows = try request.buildVoiceCloneRows(std.testing.allocator, .{
        .n_vq = 2,
        .audio_pad_token_id = 1024,
        .audio_start_token_id = 6,
        .audio_end_token_id = 7,
        .audio_user_slot_token_id = 8,
    }, templates, &prompt_audio_codes, &.{21});
    defer rows.deinit();

    const tts = bundle.TtsSummary{
        .files = .{ .prefill = "prefill.onnx" },
        .onnx = .{ .prefill_output_names = &.{ "global_hidden", "present_key_0" } },
    };
    var invocation = try buildPrefillInvocation(std.testing.allocator, tts, rows);
    defer invocation.deinit();

    try std.testing.expectEqualStrings("moss_tts_prefill", invocation.stage);
    try std.testing.expectEqualStrings("prefill.onnx", invocation.model_file);
    try std.testing.expect(invocation.hasConcreteInputs());
    try std.testing.expectEqual(@as(usize, 2), invocation.inputCount());
    try std.testing.expectEqualSlices(usize, &.{ 1, rows.sequenceLength(), rows.row_width }, invocation.inputs[0].shape);
    try std.testing.expectEqual(@as(usize, rows.sequenceLength() * rows.row_width), invocation.inputs[0].buffer.len());
}
