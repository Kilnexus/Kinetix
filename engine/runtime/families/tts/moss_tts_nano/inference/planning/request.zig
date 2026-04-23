const std = @import("std");
const bundle = @import("../../bundle/index.zig");
const manifest = @import("../../bundle/manifest.zig");
const graph_invocation = @import("../graph/invocation.zig");
const request_rows = @import("../request.zig");
const token_plan = @import("token.zig");

pub const Plan = struct {
    allocator: std.mem.Allocator,
    prefill_sequence_lengths: []usize,
    row_widths: []usize,
    prompt_audio_frame_counts: []usize,
    prefill_input_counts: []usize,
    prefill_output_counts: []usize,
    prefill_concrete_input_counts: []usize,
    ready: bool,
    graph_invocations_ready: bool,

    pub fn deinit(self: *const Plan) void {
        self.allocator.free(self.prefill_sequence_lengths);
        self.allocator.free(self.row_widths);
        self.allocator.free(self.prompt_audio_frame_counts);
        self.allocator.free(self.prefill_input_counts);
        self.allocator.free(self.prefill_output_counts);
        self.allocator.free(self.prefill_concrete_input_counts);
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    summary: manifest.Summary,
    tts: bundle.TtsSummary,
    tokens: token_plan.Plan,
) !Plan {
    const sequence_lengths = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(sequence_lengths);
    const row_widths = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(row_widths);
    const prompt_audio_frame_counts = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(prompt_audio_frame_counts);
    const prefill_input_counts = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(prefill_input_counts);
    const prefill_output_counts = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(prefill_output_counts);
    const prefill_concrete_input_counts = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(prefill_concrete_input_counts);

    if (!tokens.tokenizer_loaded or summary.default_prompt_audio_codes.len == 0) {
        @memset(sequence_lengths, 0);
        @memset(row_widths, 0);
        @memset(prompt_audio_frame_counts, summary.default_prompt_audio_codes.len);
        @memset(prefill_input_counts, 0);
        @memset(prefill_output_counts, 0);
        @memset(prefill_concrete_input_counts, 0);
        return .{
            .allocator = allocator,
            .prefill_sequence_lengths = sequence_lengths,
            .row_widths = row_widths,
            .prompt_audio_frame_counts = prompt_audio_frame_counts,
            .prefill_input_counts = prefill_input_counts,
            .prefill_output_counts = prefill_output_counts,
            .prefill_concrete_input_counts = prefill_concrete_input_counts,
            .ready = false,
            .graph_invocations_ready = false,
        };
    }

    var all_graph_invocations_ready = true;
    for (
        tokens.chunk_token_ids,
        sequence_lengths,
        row_widths,
        prompt_audio_frame_counts,
        prefill_input_counts,
        prefill_output_counts,
        prefill_concrete_input_counts,
    ) |ids, *sequence_len, *row_width, *prompt_frames, *prefill_inputs, *prefill_outputs, *concrete_inputs| {
        var rows = try request_rows.buildVoiceCloneRows(
            allocator,
            summary.tts_config,
            summary.prompt_templates,
            summary.default_prompt_audio_codes,
            ids,
        );
        defer rows.deinit();
        sequence_len.* = rows.sequenceLength();
        row_width.* = rows.row_width;
        prompt_frames.* = rows.prompt_audio_frame_count;

        var prefill = graph_invocation.buildPrefillInvocation(allocator, tts, rows) catch |err| switch (err) {
            error.MissingPrefillGraph, error.MissingPrefillOutputs => {
                all_graph_invocations_ready = false;
                prefill_inputs.* = 0;
                prefill_outputs.* = 0;
                concrete_inputs.* = 0;
                continue;
            },
            else => return err,
        };
        defer prefill.deinit();
        prefill_inputs.* = prefill.inputCount();
        prefill_outputs.* = prefill.outputCount();
        concrete_inputs.* = countConcreteInputs(prefill.inputs);
    }

    return .{
        .allocator = allocator,
        .prefill_sequence_lengths = sequence_lengths,
        .row_widths = row_widths,
        .prompt_audio_frame_counts = prompt_audio_frame_counts,
        .prefill_input_counts = prefill_input_counts,
        .prefill_output_counts = prefill_output_counts,
        .prefill_concrete_input_counts = prefill_concrete_input_counts,
        .ready = true,
        .graph_invocations_ready = all_graph_invocations_ready,
    };
}

fn countConcreteInputs(inputs: []const graph_invocation.bindings.TensorBinding) usize {
    var total: usize = 0;
    for (inputs) |input| {
        if (input.hasConcreteBuffer()) total += 1;
    }
    return total;
}
