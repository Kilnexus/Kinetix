const std = @import("std");

pub const default_max_estimated_tokens_per_chunk: usize = 75;

pub const Options = struct {
    max_estimated_tokens_per_chunk: usize = default_max_estimated_tokens_per_chunk,
};

pub const PreparedText = struct {
    allocator: std.mem.Allocator,
    normalized: []u8,
    chunks: []const []u8,
    estimated_tokens: []usize,
    uses_estimated_token_budget: bool = true,

    pub fn deinit(self: *PreparedText) void {
        for (self.chunks) |chunk| self.allocator.free(chunk);
        self.allocator.free(self.chunks);
        self.allocator.free(self.estimated_tokens);
        self.allocator.free(self.normalized);
        self.* = undefined;
    }
};

pub fn prepare(allocator: std.mem.Allocator, input: []const u8, options: Options) !PreparedText {
    const normalized = try normalizeText(allocator, input);
    errdefer allocator.free(normalized);

    var chunks = std.ArrayListUnmanaged([]u8).empty;
    errdefer {
        for (chunks.items) |chunk| allocator.free(chunk);
        chunks.deinit(allocator);
    }
    var estimated_tokens = std.ArrayListUnmanaged(usize).empty;
    errdefer estimated_tokens.deinit(allocator);

    try splitIntoChunks(
        allocator,
        normalized,
        @max(@as(usize, 1), options.max_estimated_tokens_per_chunk),
        &chunks,
        &estimated_tokens,
    );

    return .{
        .allocator = allocator,
        .normalized = normalized,
        .chunks = try chunks.toOwnedSlice(allocator),
        .estimated_tokens = try estimated_tokens.toOwnedSlice(allocator),
        .uses_estimated_token_budget = true,
    };
}

fn normalizeText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var index: usize = 0;
    var previous_space = true;
    while (index < input.len) {
        const len = utf8LenAt(input, index);
        const slice = input[index .. index + len];
        index += len;

        if (isWhitespace(slice)) {
            if (!previous_space and out.items.len != 0) {
                try out.append(allocator, ' ');
                previous_space = true;
            }
            continue;
        }

        try out.appendSlice(allocator, slice);
        previous_space = false;
    }

    while (out.items.len != 0 and out.items[out.items.len - 1] == ' ') {
        _ = out.pop();
    }
    if (out.items.len == 0) return error.EmptyTextInput;
    if (!endsWithSentencePunctuation(out.items)) {
        try out.appendSlice(allocator, if (containsCjk(out.items)) "。" else ".");
    }

    return try out.toOwnedSlice(allocator);
}

fn splitIntoChunks(
    allocator: std.mem.Allocator,
    normalized: []const u8,
    max_estimated_tokens: usize,
    chunks: *std.ArrayListUnmanaged([]u8),
    estimated_tokens: *std.ArrayListUnmanaged(usize),
) !void {
    var start: usize = 0;
    var index: usize = 0;
    var last_boundary: ?usize = null;
    var current_tokens: usize = 0;

    while (index < normalized.len) {
        const len = utf8LenAt(normalized, index);
        const token_count = estimatedTokenCost(normalized[index .. index + len]);
        current_tokens += token_count;
        index += len;

        if (isSplitBoundary(normalized[index - len .. index])) {
            last_boundary = index;
        }

        if (current_tokens >= max_estimated_tokens) {
            const end = last_boundary orelse index;
            try appendChunk(allocator, normalized[start..end], chunks, estimated_tokens);
            start = trimLeadingSpaces(normalized, end);
            index = start;
            last_boundary = null;
            current_tokens = 0;
        }
    }

    if (start < normalized.len) {
        try appendChunk(allocator, normalized[start..], chunks, estimated_tokens);
    }
}

fn appendChunk(
    allocator: std.mem.Allocator,
    raw: []const u8,
    chunks: *std.ArrayListUnmanaged([]u8),
    estimated_tokens: *std.ArrayListUnmanaged(usize),
) !void {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return;
    const chunk = try allocator.dupe(u8, trimmed);
    errdefer allocator.free(chunk);
    try chunks.append(allocator, chunk);
    try estimated_tokens.append(allocator, estimateTokens(trimmed));
}

pub fn estimateTokens(text: []const u8) usize {
    var total: usize = 0;
    var index: usize = 0;
    while (index < text.len) {
        const len = utf8LenAt(text, index);
        total += estimatedTokenCost(text[index .. index + len]);
        index += len;
    }
    return total;
}

fn estimatedTokenCost(slice: []const u8) usize {
    if (slice.len == 1 and slice[0] == ' ') return 0;
    return 1;
}

fn trimLeadingSpaces(text: []const u8, offset: usize) usize {
    var index = offset;
    while (index < text.len and text[index] == ' ') index += 1;
    return index;
}

fn utf8LenAt(text: []const u8, index: usize) usize {
    return std.unicode.utf8ByteSequenceLength(text[index]) catch 1;
}

fn isWhitespace(slice: []const u8) bool {
    if (slice.len == 1) {
        return switch (slice[0]) {
            ' ', '\t', '\n', '\r' => true,
            else => false,
        };
    }
    return std.mem.eql(u8, slice, "\u{3000}");
}

fn isSplitBoundary(slice: []const u8) bool {
    if (slice.len == 1) {
        return switch (slice[0]) {
            '.', '!', '?', ';', ',', ':' => true,
            else => false,
        };
    }
    return std.mem.eql(u8, slice, "。") or
        std.mem.eql(u8, slice, "！") or
        std.mem.eql(u8, slice, "？") or
        std.mem.eql(u8, slice, "；") or
        std.mem.eql(u8, slice, "，") or
        std.mem.eql(u8, slice, "：");
}

fn endsWithSentencePunctuation(text: []const u8) bool {
    if (text.len == 0) return false;
    var last_start: usize = 0;
    var index: usize = 0;
    while (index < text.len) {
        last_start = index;
        index += utf8LenAt(text, index);
    }
    const last = text[last_start..text.len];
    return isSplitBoundary(last);
}

fn containsCjk(text: []const u8) bool {
    var index: usize = 0;
    while (index < text.len) {
        const len = utf8LenAt(text, index);
        const cp = std.unicode.utf8Decode(text[index .. index + len]) catch {
            index += len;
            continue;
        };
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x3040 and cp <= 0x30FF) or
            (cp >= 0xAC00 and cp <= 0xD7AF))
        {
            return true;
        }
        index += len;
    }
    return false;
}

test "moss tts frontend normalizes whitespace and punctuation" {
    var prepared = try prepare(std.testing.allocator, "  hello   moss  ", .{});
    defer prepared.deinit();

    try std.testing.expectEqualStrings("hello moss.", prepared.normalized);
    try std.testing.expectEqual(@as(usize, 1), prepared.chunks.len);
    try std.testing.expectEqualStrings("hello moss.", prepared.chunks[0]);
}

test "moss tts frontend appends cjk punctuation and splits long text" {
    var prepared = try prepare(std.testing.allocator, "你好 世界。第二句很长很长", .{
        .max_estimated_tokens_per_chunk = 5,
    });
    defer prepared.deinit();

    try std.testing.expect(std.mem.endsWith(u8, prepared.normalized, "。"));
    try std.testing.expect(prepared.chunks.len >= 2);
}
