const std = @import("std");
const manifest = @import("../../bundle/manifest.zig");
const request_rows = @import("../request.zig");
const token_plan = @import("token.zig");

pub const Plan = struct {
    allocator: std.mem.Allocator,
    prefill_sequence_lengths: []usize,
    row_widths: []usize,
    prompt_audio_frame_counts: []usize,
    ready: bool,

    pub fn deinit(self: *const Plan) void {
        self.allocator.free(self.prefill_sequence_lengths);
        self.allocator.free(self.row_widths);
        self.allocator.free(self.prompt_audio_frame_counts);
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    summary: manifest.Summary,
    tokens: token_plan.Plan,
) !Plan {
    const sequence_lengths = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(sequence_lengths);
    const row_widths = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(row_widths);
    const prompt_audio_frame_counts = try allocator.alloc(usize, tokens.chunk_token_ids.len);
    errdefer allocator.free(prompt_audio_frame_counts);

    if (!tokens.tokenizer_loaded or summary.default_prompt_audio_codes.len == 0) {
        @memset(sequence_lengths, 0);
        @memset(row_widths, 0);
        @memset(prompt_audio_frame_counts, summary.default_prompt_audio_codes.len);
        return .{
            .allocator = allocator,
            .prefill_sequence_lengths = sequence_lengths,
            .row_widths = row_widths,
            .prompt_audio_frame_counts = prompt_audio_frame_counts,
            .ready = false,
        };
    }

    for (tokens.chunk_token_ids, sequence_lengths, row_widths, prompt_audio_frame_counts) |ids, *sequence_len, *row_width, *prompt_frames| {
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
    }

    return .{
        .allocator = allocator,
        .prefill_sequence_lengths = sequence_lengths,
        .row_widths = row_widths,
        .prompt_audio_frame_counts = prompt_audio_frame_counts,
        .ready = true,
    };
}
