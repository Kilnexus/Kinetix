const std = @import("std");

const TokenizerConfig = struct {
    do_lower_case: bool = true,
    model_max_length: usize = 512,
};

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    vocab: std.StringHashMapUnmanaged(u32),
    id_to_token: []const []const u8,
    do_lower_case: bool,
    unk_token: []const u8,
    cls_token: []const u8,
    sep_token: []const u8,
    pad_token: []const u8,
    mask_token: []const u8,

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    pub fn loadFromModelDir(backing_allocator: std.mem.Allocator, model_dir: []const u8) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(backing_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const vocab_path = try std.fs.path.join(allocator, &.{ model_dir, "vocab.txt" });
        defer allocator.free(vocab_path);
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "tokenizer_config.json" });
        defer allocator.free(config_path);

        const vocab_bytes = try std.fs.cwd().readFileAlloc(allocator, vocab_path, 1024 * 1024);
        var vocab = std.StringHashMapUnmanaged(u32).empty;
        var id_to_token_list = std.ArrayListUnmanaged([]const u8).empty;
        errdefer id_to_token_list.deinit(allocator);

        var line_it = std.mem.splitScalar(u8, vocab_bytes, '\n');
        var token_id: u32 = 0;
        while (line_it.next()) |raw_line| {
            const token = std.mem.trimRight(u8, raw_line, "\r");
            const owned = try allocator.dupe(u8, token);
            try vocab.put(allocator, owned, token_id);
            try id_to_token_list.append(allocator, owned);
            token_id += 1;
        }

        const tokenizer_config = loadTokenizerConfig(allocator, config_path) catch TokenizerConfig{};

        return .{
            .arena = arena,
            .vocab = vocab,
            .id_to_token = try id_to_token_list.toOwnedSlice(allocator),
            .do_lower_case = tokenizer_config.do_lower_case,
            .unk_token = "[UNK]",
            .cls_token = "[CLS]",
            .sep_token = "[SEP]",
            .pad_token = "[PAD]",
            .mask_token = "[MASK]",
        };
    }

    pub fn encodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids = std.ArrayListUnmanaged(u32).empty;
        defer ids.deinit(allocator);

        var index: usize = 0;
        while (index < text.len) {
            if (self.matchSpecialToken(text[index..])) |special_token| {
                try ids.append(allocator, self.vocab.get(special_token) orelse return error.UnknownTokenId);
                index += special_token.len;
                continue;
            }

            const next_special = self.findNextSpecialTokenStart(text[index..]) orelse text.len - index;
            var basic = std.ArrayListUnmanaged([]const u8).empty;
            defer basic.deinit(allocator);
            try self.basicTokenize(allocator, text[index .. index + next_special], &basic);

            for (basic.items) |token| {
                defer allocator.free(token);
                try self.wordpieceTokenize(allocator, token, &ids);
            }
            index += next_special;
        }

        return ids.toOwnedSlice(allocator);
    }

    pub fn decodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        defer output.deinit(allocator);

        for (ids, 0..) |id, idx| {
            if (id >= self.id_to_token.len) return error.UnknownTokenId;
            const token = self.id_to_token[id];
            const is_subword = std.mem.startsWith(u8, token, "##");

            if (idx != 0 and !is_subword) {
                try output.append(allocator, ' ');
            }

            if (is_subword) {
                try output.appendSlice(allocator, token[2..]);
            } else {
                try output.appendSlice(allocator, token);
            }
        }

        return output.toOwnedSlice(allocator);
    }

    pub fn tokenForId(self: *const Tokenizer, id: u32) ?[]const u8 {
        if (id >= self.id_to_token.len) return null;
        return self.id_to_token[id];
    }

    pub fn idForToken(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.vocab.get(token);
    }

    fn basicTokenize(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
        output: *std.ArrayListUnmanaged([]const u8),
    ) !void {
        var current = std.ArrayListUnmanaged(u8).empty;
        defer current.deinit(allocator);

        for (text) |byte| {
            if (isWhitespace(byte)) {
                try flushCurrentToken(allocator, output, &current);
                continue;
            }

            if (isAsciiPunctuation(byte)) {
                try flushCurrentToken(allocator, output, &current);
                const punct = try allocator.alloc(u8, 1);
                punct[0] = byte;
                try output.append(allocator, punct);
                continue;
            }

            try current.append(allocator, if (self.do_lower_case) std.ascii.toLower(byte) else byte);
        }

        try flushCurrentToken(allocator, output, &current);
    }

    fn wordpieceTokenize(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        token: []const u8,
        output: *std.ArrayListUnmanaged(u32),
    ) !void {
        if (token.len == 0) return;
        if (token.len > 100) {
            try output.append(allocator, self.vocab.get(self.unk_token) orelse return error.UnknownTokenId);
            return;
        }

        var start: usize = 0;
        while (start < token.len) {
            var end = token.len;
            var matched_id: ?u32 = null;
            var matched_end: usize = start;

            while (end > start) : (end -= 1) {
                const candidate = if (start == 0)
                    token[start..end]
                else
                    try std.fmt.allocPrint(allocator, "##{s}", .{token[start..end]});
                defer if (start != 0) allocator.free(candidate);

                if (self.vocab.get(candidate)) |id| {
                    matched_id = id;
                    matched_end = end;
                    break;
                }
            }

            if (matched_id) |id| {
                try output.append(allocator, id);
                start = matched_end;
                continue;
            }

            try output.append(allocator, self.vocab.get(self.unk_token) orelse return error.UnknownTokenId);
            return;
        }
    }

    fn matchSpecialToken(self: *const Tokenizer, text: []const u8) ?[]const u8 {
        const special_tokens = [_][]const u8{
            self.mask_token,
            self.cls_token,
            self.sep_token,
            self.pad_token,
            self.unk_token,
        };
        for (special_tokens) |token| {
            if (std.mem.startsWith(u8, text, token)) return token;
        }
        return null;
    }

    fn findNextSpecialTokenStart(self: *const Tokenizer, text: []const u8) ?usize {
        var best: ?usize = null;
        const special_tokens = [_][]const u8{
            self.mask_token,
            self.cls_token,
            self.sep_token,
            self.pad_token,
            self.unk_token,
        };

        for (special_tokens) |token| {
            if (std.mem.indexOf(u8, text, token)) |idx| {
                if (best == null or idx < best.?) best = idx;
            }
        }
        return best;
    }

};

fn loadTokenizerConfig(allocator: std.mem.Allocator, path: []const u8) !TokenizerConfig {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024);
    return try std.json.parseFromSliceLeaky(TokenizerConfig, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
}

fn flushCurrentToken(
    allocator: std.mem.Allocator,
    output: *std.ArrayListUnmanaged([]const u8),
    current: *std.ArrayListUnmanaged(u8),
) !void {
    if (current.items.len == 0) return;
    const owned = try current.toOwnedSlice(allocator);
    current.* = .empty;
    try output.append(allocator, owned);
}

fn isWhitespace(byte: u8) bool {
    return std.ascii.isWhitespace(byte);
}

fn isAsciiPunctuation(byte: u8) bool {
    return switch (byte) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

test "wordpiece tokenizer encodes and decodes bert-base samples" {
    const testing = std.testing;

    var tokenizer = try Tokenizer.loadFromModelDir(testing.allocator, "models/bert-base-uncased");
    defer tokenizer.deinit();

    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "Playing world!");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 2652, 2088, 999 }, ids);
    }

    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "Tokenizer");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 19204, 17629 }, ids);
    }

    {
        const ids = try tokenizer.encodeAlloc(testing.allocator, "Hello [MASK] world");
        defer testing.allocator.free(ids);
        try testing.expectEqualSlices(u32, &[_]u32{ 7592, 103, 2088 }, ids);
    }

    const text = try tokenizer.decodeAlloc(testing.allocator, &[_]u32{ 101, 7592, 2088, 999, 102 });
    defer testing.allocator.free(text);
    try testing.expectEqualStrings("[CLS] hello world ! [SEP]", text);
}
