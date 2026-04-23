const std = @import("std");
const frontend = @import("../../frontend/index.zig");
const sentencepiece = @import("../../tokenizer/sentencepiece.zig");

pub const Plan = struct {
    allocator: std.mem.Allocator,
    chunk_token_counts: []usize,
    chunk_token_ids: []const []const u32,
    tokenizer_loaded: bool,
    tokenizer_summary: sentencepiece.Summary,
    uses_estimated_token_budget: bool,

    pub fn deinit(self: *const Plan) void {
        for (self.chunk_token_ids) |ids| self.allocator.free(ids);
        self.allocator.free(self.chunk_token_ids);
        self.allocator.free(self.chunk_token_counts);
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    tokenizer: ?*const sentencepiece.Model,
    prepared_text: frontend.text.PreparedText,
) !Plan {
    const counts = try allocator.alloc(usize, prepared_text.chunks.len);
    errdefer allocator.free(counts);
    const chunk_token_ids = try allocator.alloc([]const u32, prepared_text.chunks.len);
    var token_id_chunks_filled: usize = 0;
    errdefer {
        for (chunk_token_ids[0..token_id_chunks_filled]) |ids| allocator.free(ids);
        allocator.free(chunk_token_ids);
    }

    if (tokenizer) |model| {
        for (prepared_text.chunks, counts, chunk_token_ids) |chunk, *count, *ids_slot| {
            const ids = try model.encodeAlloc(allocator, chunk);
            ids_slot.* = ids;
            token_id_chunks_filled += 1;
            count.* = ids.len;
        }
        return .{
            .allocator = allocator,
            .chunk_token_counts = counts,
            .chunk_token_ids = chunk_token_ids,
            .tokenizer_loaded = true,
            .tokenizer_summary = model.summary(),
            .uses_estimated_token_budget = false,
        };
    }

    @memcpy(counts, prepared_text.estimated_tokens);
    for (chunk_token_ids) |*ids| ids.* = &.{};
    return .{
        .allocator = allocator,
        .chunk_token_counts = counts,
        .chunk_token_ids = chunk_token_ids,
        .tokenizer_loaded = false,
        .tokenizer_summary = .{},
        .uses_estimated_token_budget = true,
    };
}
