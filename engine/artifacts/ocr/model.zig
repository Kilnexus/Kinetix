const std = @import("std");
const fs_compat = @import("engine_fs_compat");

const magic = [_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 };

pub const TensorBlob = struct {
    name: []u8,
    shape: []usize,
    values: []f32,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayList(TensorBlob),

    pub fn init(allocator: std.mem.Allocator) Model {
        return .{
            .allocator = allocator,
            .tensors = .empty,
        };
    }

    pub fn deinit(self: *Model) void {
        for (self.tensors.items) |tensor| {
            self.allocator.free(tensor.name);
            self.allocator.free(tensor.shape);
            self.allocator.free(tensor.values);
        }
        self.tensors.deinit(self.allocator);
    }

    pub fn tensorCount(self: *const Model) usize {
        return self.tensors.items.len;
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Model {
        const file = if (std.fs.path.isAbsolute(path))
            try fs_compat.openFileAbsolute(path, .{})
        else
            try fs_compat.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size > std.math.maxInt(usize)) return error.FileTooLarge;

        const byte_len: usize = @intCast(stat.size);
        const bytes = try allocator.alloc(u8, byte_len);
        defer allocator.free(bytes);
        _ = try file.readAll(bytes);

        if (bytes.len < magic.len) return error.InvalidFormat;
        if (!std.mem.eql(u8, bytes[0..magic.len], &magic)) return error.InvalidMagic;

        var cursor: usize = magic.len;

        const tensor_count = try readIntAt(u32, bytes, &cursor);
        var model = Model.init(allocator);
        errdefer model.deinit();

        for (0..tensor_count) |_| {
            const name_len = try readIntAt(u16, bytes, &cursor);
            if (name_len == 0) return error.InvalidFormat;

            const name = try allocator.alloc(u8, name_len);
            errdefer allocator.free(name);
            try readNoEof(bytes, &cursor, name);

            const ndim = try readByteAt(bytes, &cursor);
            if (ndim == 0) return error.InvalidFormat;

            const shape = try allocator.alloc(usize, ndim);
            errdefer allocator.free(shape);

            var expected_len: usize = 1;
            for (0..ndim) |i| {
                const dim = @as(usize, try readIntAt(u32, bytes, &cursor));
                if (dim == 0) return error.InvalidFormat;
                shape[i] = dim;
                expected_len = try std.math.mul(usize, expected_len, dim);
            }

            const value_len = @as(usize, try readIntAt(u32, bytes, &cursor));
            if (value_len != expected_len) return error.InvalidFormat;

            const values = try allocator.alloc(f32, value_len);
            errdefer allocator.free(values);
            for (0..value_len) |i| {
                values[i] = @bitCast(try readIntAt(u32, bytes, &cursor));
            }

            try model.tensors.append(allocator, .{
                .name = name,
                .shape = shape,
                .values = values,
            });
        }

        return model;
    }
};

fn readNoEof(bytes: []const u8, cursor: *usize, dest: []u8) !void {
    if (bytes.len - cursor.* < dest.len) return error.EndOfStream;
    @memcpy(dest, bytes[cursor.* ..][0..dest.len]);
    cursor.* += dest.len;
}

fn readByteAt(bytes: []const u8, cursor: *usize) !u8 {
    if (cursor.* >= bytes.len) return error.EndOfStream;
    const value = bytes[cursor.*];
    cursor.* += 1;
    return value;
}

fn readIntAt(comptime T: type, bytes: []const u8, cursor: *usize) !T {
    const byte_count = @divExact(@bitSizeOf(T), 8);
    if (bytes.len - cursor.* < byte_count) return error.EndOfStream;
    const value = std.mem.readInt(T, bytes[cursor.* ..][0..byte_count], .little);
    cursor.* += byte_count;
    return value;
}

test "ocr model parser loads custom format" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("demo.swm", .{});
    defer file.close();

    var file_writer = file.writer(&.{});
    const writer = &file_writer.interface;
    try writer.writeAll(&magic);
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u16, 2, .little);
    try writer.writeAll("fc");
    try writer.writeByte(1);
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u32, @bitCast(@as(f32, 1.0)), .little);
    try writer.flush();

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.swm");
    defer std.testing.allocator.free(path);

    var model = try Model.loadFromFile(std.testing.allocator, path);
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 1), model.tensorCount());
}
