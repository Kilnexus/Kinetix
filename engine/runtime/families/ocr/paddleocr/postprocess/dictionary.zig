const std = @import("std");

const io = std.Options.debug_io;

pub const Dictionary = struct {
    arena: std.heap.ArenaAllocator,
    tokens: []const []const u8,

    pub fn deinit(self: *Dictionary) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn loadFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !Dictionary {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();
    const allocator = arena.allocator();

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    var tokens = std.ArrayListUnmanaged([]const u8).empty;
    var iterator = std.mem.splitScalar(u8, bytes, '\n');
    while (iterator.next()) |raw_line| {
        const line = stripTrailingCr(raw_line);
        if (line.len == 0) continue;
        try tokens.append(allocator, line);
    }

    return .{
        .arena = arena,
        .tokens = try tokens.toOwnedSlice(allocator),
    };
}

fn stripTrailingCr(line: []const u8) []const u8 {
    if (line.len != 0 and line[line.len - 1] == '\r') return line[0 .. line.len - 1];
    return line;
}

test "paddleocr dictionary loader preserves space token" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("dict.txt", .{});
    defer file.close();
    var writer_impl = file.writer(io, &.{});
    const writer = &writer_impl.interface;
    try writer.writeAll("a\r\nb\n \n");
    try writer.flush();
    const path = try tmp.dir.realPathFileAlloc(io, "dict.txt", std.testing.allocator);
    defer std.testing.allocator.free(path);

    var dictionary = try loadFromFile(std.testing.allocator, path);
    defer dictionary.deinit();
    try std.testing.expectEqual(@as(usize, 3), dictionary.tokens.len);
    try std.testing.expectEqualStrings(" ", dictionary.tokens[2]);
}
