const std = @import("std");
const bundle = @import("../../bundle/index.zig");
const graph_invocation = @import("../graph/invocation.zig");

pub const Plan = struct {
    decode_step_ready: bool,
    decode_step_input_count: usize,
    decode_step_output_count: usize,
    codec_decode_ready: bool,
    codec_decode_input_count: usize,
    codec_decode_output_count: usize,
    template_count: usize,

    pub fn ready(self: Plan) bool {
        return self.decode_step_ready and self.codec_decode_ready;
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    tts: bundle.TtsSummary,
    codec: bundle.CodecConfig,
    row_width: usize,
    max_new_frames: usize,
) !Plan {
    var plan = Plan{
        .decode_step_ready = false,
        .decode_step_input_count = 0,
        .decode_step_output_count = 0,
        .codec_decode_ready = false,
        .codec_decode_input_count = 0,
        .codec_decode_output_count = 0,
        .template_count = 0,
    };

    const cache_input_count = if (tts.onnx.decode_input_names.len > 2) tts.onnx.decode_input_names.len - 2 else 0;
    var decode_step = graph_invocation.buildDecodeStepTemplate(
        allocator,
        tts,
        row_width,
        cache_input_count,
    ) catch |err| switch (err) {
        error.MissingDecodeGraph, error.MissingDecodeOutputs => null,
        else => return err,
    };
    if (decode_step) |*invocation| {
        defer invocation.deinit();
        plan.decode_step_ready = true;
        plan.decode_step_input_count = invocation.inputCount();
        plan.decode_step_output_count = invocation.outputCount();
        plan.template_count += 1;
    }

    var codec_decode = graph_invocation.buildCodecDecodeTemplate(
        allocator,
        codec,
        @max(@as(usize, 1), max_new_frames),
    ) catch |err| switch (err) {
        error.MissingCodecDecodeGraph, error.MissingCodecDecodeOutputs => null,
        else => return err,
    };
    if (codec_decode) |*invocation| {
        defer invocation.deinit();
        plan.codec_decode_ready = true;
        plan.codec_decode_input_count = invocation.inputCount();
        plan.codec_decode_output_count = invocation.outputCount();
        plan.template_count += 1;
    }

    return plan;
}

test "moss graph planning validates decode and codec templates" {
    const tts = bundle.TtsSummary{
        .files = .{ .decode_step = "decode.onnx" },
        .onnx = .{
            .decode_input_names = &.{ "input_ids", "past_valid_lengths", "past_key_0", "past_value_0" },
            .decode_output_names = &.{ "global_hidden", "present_key_0", "present_value_0" },
        },
    };
    const codec = bundle.CodecConfig{
        .num_quantizers = 16,
        .files = .{ .decode_full = "codec_decode.onnx" },
        .onnx = .{
            .decode_input_names = &.{ "audio_codes", "audio_code_lengths" },
            .decode_output_names = &.{ "audio", "audio_lengths" },
        },
    };

    const plan = try build(std.testing.allocator, tts, codec, 17, 375);
    try std.testing.expect(plan.ready());
    try std.testing.expectEqual(@as(usize, 4), plan.decode_step_input_count);
    try std.testing.expectEqual(@as(usize, 3), plan.decode_step_output_count);
    try std.testing.expectEqual(@as(usize, 2), plan.codec_decode_input_count);
    try std.testing.expectEqual(@as(usize, 2), plan.template_count);
}
