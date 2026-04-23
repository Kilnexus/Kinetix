const std = @import("std");
const manifest = @import("../bundle/manifest.zig");

pub const RequestRows = struct {
    allocator: std.mem.Allocator,
    input_ids: []const []const i32,
    attention_mask: []const []const i32,
    row_width: usize,
    prompt_audio_frame_count: usize,
    text_token_count: usize,

    pub fn deinit(self: *RequestRows) void {
        freeMatrix(self.allocator, self.input_ids);
        freeMatrix(self.allocator, self.attention_mask);
        self.* = undefined;
    }

    pub fn sequenceLength(self: RequestRows) usize {
        return self.input_ids.len;
    }
};

pub fn buildVoiceCloneRows(
    allocator: std.mem.Allocator,
    config: manifest.TtsConfig,
    templates: manifest.PromptTemplates,
    prompt_audio_codes: []const []const i32,
    text_token_ids: []const u32,
) !RequestRows {
    if (config.n_vq == 0) return error.InvalidMossTtsConfig;
    const row_width = config.n_vq + 1;
    const sequence_len =
        templates.user_prompt_prefix_token_ids.len +
        1 +
        prompt_audio_codes.len +
        1 +
        templates.user_prompt_after_reference_token_ids.len +
        text_token_ids.len +
        templates.assistant_prompt_prefix_token_ids.len +
        1;

    const input_ids = try allocator.alloc([]const i32, sequence_len);
    var rows_filled: usize = 0;
    errdefer {
        for (input_ids[0..rows_filled]) |row| allocator.free(row);
        allocator.free(input_ids);
    }

    try appendTextRows(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, templates.user_prompt_prefix_token_ids);
    try appendTextRow(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, config.audio_start_token_id);
    try appendAudioRows(allocator, input_ids, &rows_filled, row_width, config, prompt_audio_codes);
    try appendTextRow(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, config.audio_end_token_id);
    try appendTextRows(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, templates.user_prompt_after_reference_token_ids);
    try appendU32TextRows(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, text_token_ids);
    try appendTextRows(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, templates.assistant_prompt_prefix_token_ids);
    try appendTextRow(allocator, input_ids, &rows_filled, row_width, config.audio_pad_token_id, config.audio_start_token_id);

    const attention_mask = try allocator.alloc([]const i32, 1);
    errdefer allocator.free(attention_mask);
    attention_mask[0] = try allocator.alloc(i32, sequence_len);
    errdefer allocator.free(attention_mask[0]);
    @memset(@constCast(attention_mask[0]), 1);

    return .{
        .allocator = allocator,
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .row_width = row_width,
        .prompt_audio_frame_count = prompt_audio_codes.len,
        .text_token_count = text_token_ids.len,
    };
}

fn appendTextRows(
    allocator: std.mem.Allocator,
    rows: []const []const i32,
    filled: *usize,
    row_width: usize,
    audio_pad_token_id: i32,
    token_ids: []const i32,
) !void {
    for (token_ids) |token_id| {
        try appendTextRow(allocator, rows, filled, row_width, audio_pad_token_id, token_id);
    }
}

fn appendU32TextRows(
    allocator: std.mem.Allocator,
    rows: []const []const i32,
    filled: *usize,
    row_width: usize,
    audio_pad_token_id: i32,
    token_ids: []const u32,
) !void {
    for (token_ids) |token_id| {
        try appendTextRow(allocator, rows, filled, row_width, audio_pad_token_id, @intCast(token_id));
    }
}

fn appendTextRow(
    allocator: std.mem.Allocator,
    rows: []const []const i32,
    filled: *usize,
    row_width: usize,
    audio_pad_token_id: i32,
    token_id: i32,
) !void {
    const row = try allocator.alloc(i32, row_width);
    @memset(row, audio_pad_token_id);
    row[0] = token_id;
    @constCast(rows)[filled.*] = row;
    filled.* += 1;
}

fn appendAudioRows(
    allocator: std.mem.Allocator,
    rows: []const []const i32,
    filled: *usize,
    row_width: usize,
    config: manifest.TtsConfig,
    prompt_audio_codes: []const []const i32,
) !void {
    for (prompt_audio_codes) |code_row| {
        const row = try allocator.alloc(i32, row_width);
        @memset(row, config.audio_pad_token_id);
        row[0] = config.audio_user_slot_token_id;
        const max_codes = @min(config.n_vq, code_row.len);
        for (code_row[0..max_codes], 0..) |token, index| {
            row[index + 1] = token;
        }
        @constCast(rows)[filled.*] = row;
        filled.* += 1;
    }
}

fn freeMatrix(allocator: std.mem.Allocator, matrix: []const []const i32) void {
    for (matrix) |row| allocator.free(row);
    allocator.free(matrix);
}

test "moss tts request builder creates voice clone prefill rows" {
    const templates = manifest.PromptTemplates{
        .user_prompt_prefix_token_ids = &.{ 10, 11 },
        .user_prompt_after_reference_token_ids = &.{12},
        .assistant_prompt_prefix_token_ids = &.{ 13, 14 },
    };
    const prompt_audio_codes = [_][]const i32{
        &.{ 1, 2 },
        &.{ 3, 4 },
    };

    var rows = try buildVoiceCloneRows(std.testing.allocator, .{
        .n_vq = 2,
        .audio_pad_token_id = 1024,
        .audio_start_token_id = 6,
        .audio_end_token_id = 7,
        .audio_user_slot_token_id = 8,
    }, templates, &prompt_audio_codes, &.{ 21, 22 });
    defer rows.deinit();

    try std.testing.expectEqual(@as(usize, 11), rows.sequenceLength());
    try std.testing.expectEqual(@as(usize, 3), rows.row_width);
    try std.testing.expectEqualSlices(i32, &.{ 10, 1024, 1024 }, rows.input_ids[0]);
    try std.testing.expectEqualSlices(i32, &.{ 8, 1, 2 }, rows.input_ids[3]);
    try std.testing.expectEqualSlices(i32, &.{ 7, 1024, 1024 }, rows.input_ids[5]);
    try std.testing.expectEqualSlices(i32, &.{ 21, 1024, 1024 }, rows.input_ids[7]);
    try std.testing.expectEqual(@as(usize, 1), rows.attention_mask.len);
    try std.testing.expectEqual(@as(usize, 11), rows.attention_mask[0].len);
}
