const std = @import("std");

pub const io = std.Options.debug_io;

pub const File = struct {
    inner: std.Io.File,

    pub fn stat(self: File) std.Io.File.StatError!std.Io.File.Stat {
        return self.inner.stat(io);
    }

    pub fn close(self: File) void {
        self.inner.close(io);
    }

    pub fn reader(self: File, buffer: []u8) std.Io.File.Reader {
        return self.inner.reader(io, buffer);
    }

    pub fn writer(self: File, buffer: []u8) std.Io.File.Writer {
        return self.inner.writer(io, buffer);
    }

    pub fn readPositionalAll(self: File, buffer: []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        return self.inner.readPositionalAll(io, buffer, offset);
    }

    pub fn readAll(self: File, buffer: []u8) std.Io.File.ReadPositionalError!usize {
        return self.inner.readPositionalAll(io, buffer, 0);
    }

    pub fn readToEndAlloc(self: File, allocator: std.mem.Allocator, max_bytes: usize) std.Io.Reader.LimitedAllocError![]u8 {
        var buffer: [4096]u8 = undefined;
        var file_reader = self.inner.reader(io, &buffer);
        return file_reader.interface.allocRemaining(allocator, .limited(max_bytes));
    }

    pub fn writeStreamingAll(self: File, bytes: []const u8) std.Io.Writer.Error!void {
        return self.inner.writeStreamingAll(io, bytes);
    }
};

pub const Dir = struct {
    inner: std.Io.Dir,

    pub fn close(self: Dir) void {
        self.inner.close(io);
    }

    pub fn access(self: Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        return self.inner.access(io, sub_path, options);
    }

    pub fn openFile(self: Dir, sub_path: []const u8, options: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!File {
        return .{ .inner = try self.inner.openFile(io, sub_path, options) };
    }

    pub fn openDir(self: Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!Dir {
        return .{ .inner = try self.inner.openDir(io, sub_path, options) };
    }

    pub fn createFile(self: Dir, sub_path: []const u8, options: std.Io.Dir.CreateFileOptions) std.Io.File.OpenError!File {
        return .{ .inner = try self.inner.createFile(io, sub_path, options) };
    }

    pub fn deleteFile(self: Dir, sub_path: []const u8) std.Io.Dir.DeleteFileError!void {
        return self.inner.deleteFile(io, sub_path);
    }

    pub fn writeFile(self: Dir, options: std.Io.Dir.WriteFileOptions) std.Io.Dir.WriteFileError!void {
        return self.inner.writeFile(io, options);
    }

    pub fn readFileAlloc(self: Dir, allocator: std.mem.Allocator, sub_path: []const u8, max_bytes: usize) std.Io.Dir.ReadFileAllocError![]u8 {
        return self.inner.readFileAlloc(io, sub_path, allocator, .limited(max_bytes));
    }

    pub fn realpathAlloc(self: Dir, allocator: std.mem.Allocator, sub_path: []const u8) std.Io.Dir.RealPathFileAllocError![]u8 {
        const resolved = try self.inner.realPathFileAlloc(io, sub_path, allocator);
        defer allocator.free(resolved);
        return allocator.dupe(u8, resolved);
    }

    pub fn iterate(self: Dir) Iterator {
        return .{ .inner = self.inner.iterate() };
    }
};

pub const Iterator = struct {
    inner: std.Io.Dir.Iterator,

    pub fn next(self: *Iterator) std.Io.Dir.Iterator.Error!?std.Io.Dir.Entry {
        return self.inner.next(io);
    }
};

pub fn cwd() Dir {
    return .{ .inner = std.Io.Dir.cwd() };
}

pub fn openFileAbsolute(path: []const u8, options: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!File {
    return .{ .inner = try std.Io.Dir.openFileAbsolute(io, path, options) };
}

pub fn openDirAbsolute(path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!Dir {
    return .{ .inner = try std.Io.Dir.openDirAbsolute(io, path, options) };
}

pub fn createFileAbsolute(path: []const u8, options: std.Io.Dir.CreateFileOptions) std.Io.File.OpenError!File {
    return .{ .inner = try std.Io.Dir.createFileAbsolute(io, path, options) };
}

pub fn deleteFileAbsolute(path: []const u8) std.Io.Dir.DeleteFileError!void {
    return std.Io.Dir.deleteFileAbsolute(io, path);
}
