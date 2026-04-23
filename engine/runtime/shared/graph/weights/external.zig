const std = @import("std");
const metadata = @import("../onnx/metadata.zig");
const mapped_file = @import("mapped_file.zig");

const io = std.Options.debug_io;

pub const TensorBytes = struct {
    name: []const u8,
    elem_type: metadata.ElementType,
    dims: []const metadata.Dimension,
    bytes: []const u8,
};

const MappedDataFile = struct {
    path: []u8,
    mapped: mapped_file.MappedFile,

    fn deinit(self: *MappedDataFile, allocator: std.mem.Allocator) void {
        self.mapped.deinit();
        allocator.free(self.path);
        self.* = undefined;
    }
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    base_dir: []u8,
    files: std.ArrayListUnmanaged(MappedDataFile) = .empty,

    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) !Store {
        return .{
            .allocator = allocator,
            .base_dir = try allocator.dupe(u8, base_dir),
        };
    }

    pub fn deinit(self: *Store) void {
        for (self.files.items) |*file| file.deinit(self.allocator);
        self.files.deinit(self.allocator);
        self.allocator.free(self.base_dir);
        self.* = undefined;
    }

    pub fn tensorBytes(self: *Store, tensor: metadata.TensorInfo) !TensorBytes {
        if (!tensor.isExternal()) return error.TensorIsNotExternal;
        const location = tensor.externalValue("location") orelse return error.MissingExternalDataLocation;
        const offset = try tensor.externalOffset();
        const length = try tensor.externalLength();
        const bytes = try self.externalRange(location, offset, length);
        return .{
            .name = tensor.name,
            .elem_type = tensor.elem_type,
            .dims = tensor.dims,
            .bytes = bytes,
        };
    }

    fn externalRange(self: *Store, location: []const u8, offset: usize, length: usize) ![]const u8 {
        const file = try self.mappedFile(location);
        if (offset > file.mapped.bytes.len) return error.ExternalDataOffsetOutOfBounds;
        if (length > file.mapped.bytes.len - offset) return error.ExternalDataLengthOutOfBounds;
        return file.mapped.bytes[offset .. offset + length];
    }

    fn mappedFile(self: *Store, location: []const u8) !*MappedDataFile {
        const path = try self.resolveLocation(location);
        errdefer self.allocator.free(path);
        for (self.files.items) |*file| {
            if (std.mem.eql(u8, file.path, path)) {
                self.allocator.free(path);
                return file;
            }
        }

        const file = if (std.fs.path.isAbsolute(path))
            try std.Io.Dir.openFileAbsolute(io, path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        var mapped = try mapped_file.MappedFile.open(file);
        file.close(io);
        errdefer mapped.deinit();

        try self.files.append(self.allocator, .{
            .path = path,
            .mapped = mapped,
        });
        return &self.files.items[self.files.items.len - 1];
    }

    fn resolveLocation(self: *Store, location: []const u8) ![]u8 {
        if (std.fs.path.isAbsolute(location)) return try self.allocator.dupe(u8, location);
        return try std.fs.path.join(self.allocator, &.{ self.base_dir, location });
    }
};

test "external data store maps tensor byte ranges" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTmpFile(tmp.dir, "weights.data", &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try Store.init(std.testing.allocator, root);
    defer store.deinit();

    var entries = [_]metadata.ExternalDataEntry{
        .{
            .allocator = std.testing.allocator,
            .key = @constCast("location"),
            .value = @constCast("weights.data"),
        },
        .{
            .allocator = std.testing.allocator,
            .key = @constCast("offset"),
            .value = @constCast("2"),
        },
        .{
            .allocator = std.testing.allocator,
            .key = @constCast("length"),
            .value = @constCast("4"),
        },
    };
    const dims = [_]metadata.Dimension{.{ .value = 4 }};
    const tensor = metadata.TensorInfo{
        .allocator = std.testing.allocator,
        .name = @constCast("tensor"),
        .elem_type = .{ .raw = 2 },
        .dims = &dims,
        .data_location = 1,
        .external_data = &entries,
    };

    const bytes = try store.tensorBytes(tensor);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5 }, bytes.bytes);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, bytes: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(bytes);
}
