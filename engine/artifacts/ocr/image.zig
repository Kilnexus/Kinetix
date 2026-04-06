const std = @import("std");

pub const Image = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    channels: usize,
    pixels: []u8,

    pub fn deinit(self: *Image) void {
        self.allocator.free(self.pixels);
    }

    pub fn loadPpmFile(allocator: std.mem.Allocator, path: []const u8) !Image {
        const file = if (std.fs.path.isAbsolute(path))
            try std.fs.openFileAbsolute(path, .{})
        else
            try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size > std.math.maxInt(usize)) return error.FileTooLarge;

        const byte_len: usize = @intCast(stat.size);
        const bytes = try allocator.alloc(u8, byte_len);
        defer allocator.free(bytes);
        _ = try file.readAll(bytes);
        return try parseP6(allocator, bytes);
    }
};

fn parseP6(allocator: std.mem.Allocator, bytes: []const u8) !Image {
    var idx: usize = 0;
    const magic = try readToken(bytes, &idx);
    if (!std.mem.eql(u8, magic, "P6")) return error.UnsupportedImageFormat;

    const width_token = try readToken(bytes, &idx);
    const height_token = try readToken(bytes, &idx);
    const maxval_token = try readToken(bytes, &idx);

    const width = try std.fmt.parseInt(usize, width_token, 10);
    const height = try std.fmt.parseInt(usize, height_token, 10);
    const maxval = try std.fmt.parseInt(usize, maxval_token, 10);
    if (maxval != 255) return error.UnsupportedImageFormat;

    skipWhitespaceAndComments(bytes, &idx);
    const pixels_len = try std.math.mul(usize, try std.math.mul(usize, width, height), 3);
    if (idx + pixels_len > bytes.len) return error.InvalidImageData;

    const pixels = try allocator.alloc(u8, pixels_len);
    std.mem.copyForwards(u8, pixels, bytes[idx .. idx + pixels_len]);

    return .{
        .allocator = allocator,
        .width = width,
        .height = height,
        .channels = 3,
        .pixels = pixels,
    };
}

fn skipWhitespaceAndComments(bytes: []const u8, idx: *usize) void {
    while (idx.* < bytes.len) {
        const c = bytes[idx.*];
        if (std.ascii.isWhitespace(c)) {
            idx.* += 1;
            continue;
        }
        if (c == '#') {
            idx.* += 1;
            while (idx.* < bytes.len and bytes[idx.*] != '\n') idx.* += 1;
            continue;
        }
        break;
    }
}

fn readToken(bytes: []const u8, idx: *usize) ![]const u8 {
    skipWhitespaceAndComments(bytes, idx);
    if (idx.* >= bytes.len) return error.UnexpectedEof;

    const start = idx.*;
    while (idx.* < bytes.len) : (idx.* += 1) {
        const c = bytes[idx.*];
        if (std.ascii.isWhitespace(c) or c == '#') break;
    }
    if (start == idx.*) return error.InvalidImageHeader;
    return bytes[start..idx.*];
}

test "ppm parser loads rgb image" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("tiny.ppm", .{});
    defer file.close();
    var file_writer = file.writer(&.{});
    const writer = &file_writer.interface;
    try writer.writeAll("P6\n1 1\n255\n");
    try writer.writeAll(&[_]u8{ 1, 2, 3 });
    try writer.flush();

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "tiny.ppm");
    defer std.testing.allocator.free(path);

    var image = try Image.loadPpmFile(std.testing.allocator, path);
    defer image.deinit();

    try std.testing.expectEqual(@as(usize, 1), image.width);
    try std.testing.expectEqual(@as(usize, 1), image.height);
    try std.testing.expectEqual(@as(usize, 3), image.pixels.len);
}
