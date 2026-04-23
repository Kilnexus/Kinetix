const builtin = @import("builtin");
const std = @import("std");

const windows = std.os.windows;
const io = std.Options.debug_io;

pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) const u8,

    pub fn open(file: std.Io.File) !MappedFile {
        return switch (builtin.os.tag) {
            .windows => openWindows(file),
            else => openPosix(file),
        };
    }

    pub fn deinit(self: *MappedFile) void {
        switch (builtin.os.tag) {
            .windows => {
                const ok = UnmapViewOfFile(self.bytes.ptr);
                std.debug.assert(ok.toBool());
            },
            else => std.posix.munmap(self.bytes),
        }
    }
};

fn openPosix(file: std.Io.File) !MappedFile {
    const stat = try file.stat(io);
    const file_len = std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
    const mapped = try std.posix.mmap(
        null,
        file_len,
        std.posix.PROT.READ,
        .{ .TYPE = .SHARED },
        file.handle,
        0,
    );
    return .{ .bytes = mapped };
}

fn openWindows(file: std.Io.File) !MappedFile {
    const stat = try file.stat(io);
    const file_len = std.math.cast(usize, stat.size) orelse return error.FileTooLarge;
    const mapping = CreateFileMappingW(file.handle, null, page_readonly, 0, 0, null) orelse return error.Unexpected;
    defer windows.CloseHandle(mapping);

    const view = MapViewOfFile(mapping, file_map_read, 0, 0, file_len) orelse return error.Unexpected;
    const mapped: []align(std.heap.page_size_min) const u8 = @as([*]align(std.heap.page_size_min) const u8, @ptrCast(@alignCast(view)))[0..file_len];
    return .{ .bytes = mapped };
}

const file_map_read: windows.DWORD = 0x0004;
const page_readonly: windows.DWORD = 0x0002;

extern "kernel32" fn CreateFileMappingW(
    hFile: windows.HANDLE,
    lpFileMappingAttributes: ?*windows.SECURITY_ATTRIBUTES,
    flProtect: windows.DWORD,
    dwMaximumSizeHigh: windows.DWORD,
    dwMaximumSizeLow: windows.DWORD,
    lpName: ?windows.LPCWSTR,
) callconv(.winapi) ?windows.HANDLE;

extern "kernel32" fn MapViewOfFile(
    hFileMappingObject: windows.HANDLE,
    dwDesiredAccess: windows.DWORD,
    dwFileOffsetHigh: windows.DWORD,
    dwFileOffsetLow: windows.DWORD,
    dwNumberOfBytesToMap: usize,
) callconv(.winapi) ?*anyopaque;

extern "kernel32" fn UnmapViewOfFile(
    lpBaseAddress: *const anyopaque,
) callconv(.winapi) windows.BOOL;
