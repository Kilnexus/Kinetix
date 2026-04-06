const std = @import("std");

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
            .tensors = .{},
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

        if (bytes.len < magic.len) return error.InvalidFormat;
        if (!std.mem.eql(u8, bytes[0..magic.len], &magic)) return error.InvalidMagic;

        var stream = std.io.fixedBufferStream(bytes[magic.len..]);
        const reader = stream.reader();

        const tensor_count = try reader.readInt(u32, .little);
        var model = Model.init(allocator);
        errdefer model.deinit();

        for (0..tensor_count) |_| {
            const name_len = try reader.readInt(u16, .little);
            if (name_len == 0) return error.InvalidFormat;

            const name = try allocator.alloc(u8, name_len);
            errdefer allocator.free(name);
            try reader.readNoEof(name);

            const ndim = try reader.readByte();
            if (ndim == 0) return error.InvalidFormat;

            const shape = try allocator.alloc(usize, ndim);
            errdefer allocator.free(shape);

            var expected_len: usize = 1;
            for (0..ndim) |i| {
                const dim = @as(usize, try reader.readInt(u32, .little));
                if (dim == 0) return error.InvalidFormat;
                shape[i] = dim;
                expected_len = try std.math.mul(usize, expected_len, dim);
            }

            const value_len = @as(usize, try reader.readInt(u32, .little));
            if (value_len != expected_len) return error.InvalidFormat;

            const values = try allocator.alloc(f32, value_len);
            errdefer allocator.free(values);
            for (0..value_len) |i| {
                values[i] = @bitCast(try reader.readInt(u32, .little));
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
