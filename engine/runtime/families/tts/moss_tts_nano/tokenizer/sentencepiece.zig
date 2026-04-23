const std = @import("std");

const io = std.Options.debug_io;

pub const PieceType = enum(u32) {
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

pub const Piece = struct {
    text: []u8,
    score: f32 = 0,
    kind: PieceType = .normal,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    pieces: []Piece,
    unk_id: ?usize = null,
    bos_id: ?usize = null,
    eos_id: ?usize = null,
    pad_id: ?usize = null,
    max_piece_len: usize = 0,

    pub fn deinit(self: *Model) void {
        for (self.pieces) |piece| self.allocator.free(piece.text);
        self.allocator.free(self.pieces);
        self.* = undefined;
    }

    pub fn summary(self: *const Model) Summary {
        return .{
            .piece_count = self.pieces.len,
            .unk_id = self.unk_id,
            .bos_id = self.bos_id,
            .eos_id = self.eos_id,
            .pad_id = self.pad_id,
            .max_piece_len = self.max_piece_len,
        };
    }

    pub fn encodeAlloc(self: *const Model, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var normalized = try normalizeForSentencePiece(allocator, text);
        defer allocator.free(normalized);

        var ids = std.ArrayListUnmanaged(u32).empty;
        errdefer ids.deinit(allocator);

        var index: usize = 0;
        while (index < normalized.len) {
            if (self.findLongestPiece(normalized[index..])) |match| {
                try ids.append(allocator, @intCast(match.id));
                index += match.len;
                continue;
            }

            if (self.unk_id) |unk| {
                try ids.append(allocator, @intCast(unk));
            }
            index += utf8LenAt(normalized, index);
        }

        return try ids.toOwnedSlice(allocator);
    }

    fn findLongestPiece(self: *const Model, input: []const u8) ?PieceMatch {
        var best: ?PieceMatch = null;
        for (self.pieces, 0..) |piece, id| {
            if (piece.text.len == 0) continue;
            if (piece.kind != .normal and piece.kind != .user_defined and piece.kind != .byte) continue;
            if (!std.mem.startsWith(u8, input, piece.text)) continue;
            if (best == null or piece.text.len > best.?.len) {
                best = .{ .id = id, .len = piece.text.len };
            }
        }
        return best;
    }
};

pub const Summary = struct {
    piece_count: usize = 0,
    unk_id: ?usize = null,
    bos_id: ?usize = null,
    eos_id: ?usize = null,
    pad_id: ?usize = null,
    max_piece_len: usize = 0,
};

const PieceMatch = struct {
    id: usize,
    len: usize,
};

const Reader = struct {
    bytes: []const u8,
    index: usize = 0,

    fn eof(self: Reader) bool {
        return self.index >= self.bytes.len;
    }

    fn readVarint(self: *Reader) !u64 {
        var shift: u6 = 0;
        var value: u64 = 0;
        while (self.index < self.bytes.len and shift < 64) {
            const byte = self.bytes[self.index];
            self.index += 1;
            value |= (@as(u64, byte & 0x7f) << shift);
            if ((byte & 0x80) == 0) return value;
            shift += 7;
        }
        return error.InvalidProtobufVarint;
    }

    fn readBytes(self: *Reader, len: usize) ![]const u8 {
        if (self.index + len > self.bytes.len) return error.TruncatedProtobuf;
        const out = self.bytes[self.index .. self.index + len];
        self.index += len;
        return out;
    }

    fn skip(self: *Reader, wire_type: u3) !void {
        switch (wire_type) {
            0 => _ = try self.readVarint(),
            1 => _ = try self.readBytes(8),
            2 => {
                const len: usize = @intCast(try self.readVarint());
                _ = try self.readBytes(len);
            },
            5 => _ = try self.readBytes(4),
            else => return error.UnsupportedProtobufWireType,
        }
    }
};

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Model {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(128 * 1024 * 1024));
    defer allocator.free(bytes);
    return try parseModel(allocator, bytes);
}

pub fn parseModel(allocator: std.mem.Allocator, bytes: []const u8) !Model {
    var reader = Reader{ .bytes = bytes };
    var pieces = std.ArrayListUnmanaged(Piece).empty;
    errdefer deinitPieces(allocator, pieces.items);

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        if (field_number == 1 and wire_type == 2) {
            const len: usize = @intCast(try reader.readVarint());
            const message = try reader.readBytes(len);
            try pieces.append(allocator, try parsePiece(allocator, message));
            continue;
        }

        try reader.skip(wire_type);
    }

    if (pieces.items.len == 0) return error.EmptySentencePieceModel;

    var model = Model{
        .allocator = allocator,
        .pieces = try pieces.toOwnedSlice(allocator),
    };
    resolveSpecialIds(&model);
    return model;
}

fn parsePiece(allocator: std.mem.Allocator, bytes: []const u8) !Piece {
    var reader = Reader{ .bytes = bytes };
    var piece = Piece{
        .text = &.{},
        .score = 0,
        .kind = .normal,
    };
    var has_text = false;
    errdefer if (has_text) allocator.free(piece.text);

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidSentencePieceModel;
                const len: usize = @intCast(try reader.readVarint());
                piece.text = try allocator.dupe(u8, try reader.readBytes(len));
                has_text = true;
            },
            2 => {
                if (wire_type != 5) return error.InvalidSentencePieceModel;
                const raw = try reader.readBytes(4);
                piece.score = @bitCast(std.mem.readInt(u32, raw[0..4], .little));
            },
            3 => {
                if (wire_type != 0) return error.InvalidSentencePieceModel;
                const value: u32 = @intCast(try reader.readVarint());
                piece.kind = switch (value) {
                    1 => .normal,
                    2 => .unknown,
                    3 => .control,
                    4 => .user_defined,
                    5 => .unused,
                    6 => .byte,
                    else => .normal,
                };
            },
            else => try reader.skip(wire_type),
        }
    }

    if (!has_text) return error.InvalidSentencePieceModel;
    return piece;
}

fn resolveSpecialIds(self: *Model) void {
    for (self.pieces, 0..) |piece, id| {
        self.max_piece_len = @max(self.max_piece_len, piece.text.len);
        if (piece.kind == .unknown or std.mem.eql(u8, piece.text, "<unk>")) self.unk_id = id;
        if (std.mem.eql(u8, piece.text, "<s>")) self.bos_id = id;
        if (std.mem.eql(u8, piece.text, "</s>")) self.eos_id = id;
        if (std.mem.eql(u8, piece.text, "<pad>")) self.pad_id = id;
    }
}

fn normalizeForSentencePiece(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "\u{2581}");
    var index: usize = 0;
    var previous_space = false;
    while (index < text.len) {
        const len = utf8LenAt(text, index);
        const slice = text[index .. index + len];
        index += len;
        if (slice.len == 1 and slice[0] == ' ') {
            if (!previous_space) try out.appendSlice(allocator, "\u{2581}");
            previous_space = true;
            continue;
        }
        try out.appendSlice(allocator, slice);
        previous_space = false;
    }
    return try out.toOwnedSlice(allocator);
}

fn utf8LenAt(text: []const u8, index: usize) usize {
    return std.unicode.utf8ByteSequenceLength(text[index]) catch 1;
}

fn deinitPieces(allocator: std.mem.Allocator, pieces: []Piece) void {
    for (pieces) |piece| allocator.free(piece.text);
    allocator.free(pieces);
}

test "sentencepiece parser reads model pieces and greedy encodes text" {
    var bytes = std.ArrayList(u8).init(std.testing.allocator);
    defer bytes.deinit();

    try appendPiece(&bytes, "<unk>", -1.0, .unknown);
    try appendPiece(&bytes, "\u{2581}", -0.1, .normal);
    try appendPiece(&bytes, "hello", -0.2, .normal);
    try appendPiece(&bytes, "world", -0.3, .normal);

    var model = try parseModel(std.testing.allocator, bytes.items);
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 4), model.pieces.len);
    try std.testing.expectEqual(@as(?usize, 0), model.unk_id);

    const ids = try model.encodeAlloc(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 1, 3 }, ids);
}

fn appendPiece(bytes: *std.ArrayList(u8), text: []const u8, score: f32, kind: PieceType) !void {
    var payload = std.ArrayList(u8).init(std.testing.allocator);
    defer payload.deinit();

    try writeKey(&payload, 1, 2);
    try writeVarint(&payload, text.len);
    try payload.appendSlice(text);
    try writeKey(&payload, 2, 5);
    var score_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &score_bytes, @bitCast(score), .little);
    try payload.appendSlice(&score_bytes);
    try writeKey(&payload, 3, 0);
    try writeVarint(&payload, @intFromEnum(kind));

    try writeKey(bytes, 1, 2);
    try writeVarint(bytes, payload.items.len);
    try bytes.appendSlice(payload.items);
}

fn writeKey(bytes: *std.ArrayList(u8), field_number: u64, wire_type: u3) !void {
    try writeVarint(bytes, (field_number << 3) | wire_type);
}

fn writeVarint(bytes: *std.ArrayList(u8), raw: u64) !void {
    var value = raw;
    while (value >= 0x80) {
        try bytes.append(@intCast((value & 0x7f) | 0x80));
        value >>= 7;
    }
    try bytes.append(@intCast(value));
}
