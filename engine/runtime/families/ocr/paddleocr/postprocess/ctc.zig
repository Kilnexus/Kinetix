const std = @import("std");
const Tensor = @import("shared_graph").runtime.tensor.Tensor;

pub const DecodeOptions = struct {
    blank_id: u32 = 0,
    dictionary_base_id: u32 = 1,
};

pub const DecodedText = struct {
    allocator: std.mem.Allocator,
    text: []u8,
    token_ids: []u32,

    pub fn deinit(self: *DecodedText) void {
        self.allocator.free(self.text);
        self.allocator.free(self.token_ids);
        self.* = undefined;
    }
};

pub fn decodeIds(
    allocator: std.mem.Allocator,
    ids: []const u32,
    dictionary: []const []const u8,
    options: DecodeOptions,
) !DecodedText {
    var text = std.ArrayListUnmanaged(u8).empty;
    errdefer text.deinit(allocator);
    var tokens = std.ArrayListUnmanaged(u32).empty;
    errdefer tokens.deinit(allocator);

    var previous: ?u32 = null;
    for (ids) |id| {
        if (previous != null and previous.? == id) continue;
        previous = id;
        if (id == options.blank_id) continue;
        const piece = tokenText(dictionary, id, options.dictionary_base_id) orelse return error.DictionaryTokenNotFound;
        try text.appendSlice(allocator, piece);
        try tokens.append(allocator, id);
    }

    return .{
        .allocator = allocator,
        .text = try text.toOwnedSlice(allocator),
        .token_ids = try tokens.toOwnedSlice(allocator),
    };
}

pub fn decodeBestPathF32(
    allocator: std.mem.Allocator,
    values: []const f32,
    timesteps: usize,
    classes: usize,
    dictionary: []const []const u8,
    options: DecodeOptions,
) !DecodedText {
    if (classes == 0 or values.len != timesteps * classes) return error.ShapeMismatch;
    const ids = try allocator.alloc(u32, timesteps);
    defer allocator.free(ids);
    for (ids, 0..) |*slot, step| {
        const row = values[step * classes .. (step + 1) * classes];
        slot.* = @intCast(argMax(row));
    }
    return try decodeIds(allocator, ids, dictionary, options);
}

pub fn decodeTensorBestPath(
    allocator: std.mem.Allocator,
    logits_or_probs: Tensor,
    dictionary: []const []const u8,
    options: DecodeOptions,
) !DecodedText {
    if (logits_or_probs.buffer != .f32) return error.UnsupportedTensorDType;
    const shape = logits_or_probs.shape;
    if (shape.len == 2) {
        return try decodeBestPathF32(allocator, logits_or_probs.buffer.f32, shape[0], shape[1], dictionary, options);
    }
    if (shape.len == 3) {
        if (shape[0] != 1) return error.ShapeMismatch;
        const timesteps = shape[1];
        const classes = shape[2];
        const batch_stride = timesteps * classes;
        return try decodeBestPathF32(allocator, logits_or_probs.buffer.f32[0..batch_stride], timesteps, classes, dictionary, options);
    }
    return error.UnsupportedTensorRank;
}

fn tokenText(dictionary: []const []const u8, id: u32, base_id: u32) ?[]const u8 {
    if (id < base_id) return null;
    const index: usize = @intCast(id - base_id);
    if (index >= dictionary.len) return null;
    return dictionary[index];
}

fn argMax(values: []const f32) usize {
    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

test "paddleocr ctc decode collapses repeats and removes blank" {
    var decoded = try decodeIds(
        std.testing.allocator,
        &.{ 1, 1, 0, 2, 2 },
        &.{ "a", "b" },
        .{ .blank_id = 0, .dictionary_base_id = 1 },
    );
    defer decoded.deinit();

    try std.testing.expectEqualStrings("ab", decoded.text);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, decoded.token_ids);
}

test "paddleocr ctc decode accepts rank two logits" {
    var logits = try Tensor.fromF32(std.testing.allocator, &.{ 3, 3 }, &.{
        0.1, 0.8, 0.1,
        0.9, 0.0, 0.1,
        0.1, 0.2, 0.7,
    });
    defer logits.deinit();

    var decoded = try decodeTensorBestPath(std.testing.allocator, logits, &.{ "a", "b" }, .{});
    defer decoded.deinit();
    try std.testing.expectEqualStrings("ab", decoded.text);
}
