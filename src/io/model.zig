const std = @import("std");

const MAGIC = [_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 };

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
        return loadFromDir(allocator, std.fs.cwd(), path);
    }

    pub fn loadFromDir(allocator: std.mem.Allocator, dir: std.fs.Dir, path: []const u8) !Model {
        var file = try dir.openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        if (stat.size > std.math.maxInt(usize)) return error.FileTooLarge;

        const byte_len: usize = @intCast(stat.size);
        const bytes = try allocator.alloc(u8, byte_len);
        defer allocator.free(bytes);
        _ = try file.readAll(bytes);

        if (bytes.len < MAGIC.len) return error.InvalidFormat;
        if (!std.mem.eql(u8, bytes[0..MAGIC.len], &MAGIC)) return error.InvalidMagic;

        var stream = std.io.fixedBufferStream(bytes[MAGIC.len..]);
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
                const dim_u32 = try reader.readInt(u32, .little);
                const dim = @as(usize, dim_u32);
                if (dim == 0) return error.InvalidFormat;
                shape[i] = dim;
                expected_len = try std.math.mul(usize, expected_len, dim);
            }

            const value_len = try reader.readInt(u32, .little);
            const value_len_usize = @as(usize, value_len);
            if (value_len_usize != expected_len) return error.InvalidFormat;

            const values = try allocator.alloc(f32, value_len_usize);
            errdefer allocator.free(values);
            for (0..value_len_usize) |i| {
                const bits = try reader.readInt(u32, .little);
                values[i] = @bitCast(bits);
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

test "load custom model format" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var file = try tmp.dir.createFile("demo.swm", .{});
    defer file.close();

    var file_writer = file.writer(&.{});
    const writer = &file_writer.interface;
    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, 1, .little);

    const tensor_name = "fc.weight";
    try writer.writeInt(u16, tensor_name.len, .little);
    try writer.writeAll(tensor_name);
    try writer.writeByte(2); // ndim
    try writer.writeInt(u32, 2, .little);
    try writer.writeInt(u32, 2, .little);
    try writer.writeInt(u32, 4, .little);
    const vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    for (vals) |v| {
        try writer.writeInt(u32, @bitCast(v), .little);
    }
    try writer.flush();

    var model = try Model.loadFromDir(std.testing.allocator, tmp.dir, "demo.swm");
    defer model.deinit();

    try std.testing.expectEqual(@as(usize, 1), model.tensorCount());
    try std.testing.expectEqualStrings("fc.weight", model.tensors.items[0].name);
    try std.testing.expectEqual(@as(usize, 2), model.tensors.items[0].shape.len);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), model.tensors.items[0].values[3], 1e-6);
}
