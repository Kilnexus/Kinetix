const builtin = @import("builtin");
const std = @import("std");

const windows = std.os.windows;

pub const MappedFile = struct {
    bytes: []align(std.heap.page_size_min) const u8,

    pub fn open(file: std.fs.File) !MappedFile {
        return switch (builtin.os.tag) {
            .windows => openWindows(file),
            else => openPosix(file),
        };
    }

    pub fn deinit(self: *MappedFile) void {
        switch (builtin.os.tag) {
            .windows => {
                const ok = UnmapViewOfFile(self.bytes.ptr);
                std.debug.assert(ok != 0);
            },
            else => std.posix.munmap(self.bytes),
        }
    }
};

fn openPosix(file: std.fs.File) !MappedFile {
    const file_len = std.math.cast(usize, try file.getEndPos()) orelse return error.FileTooLarge;
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

fn openWindows(file: std.fs.File) !MappedFile {
    const file_len = std.math.cast(usize, try file.getEndPos()) orelse return error.FileTooLarge;
    const mapping = CreateFileMappingW(file.handle, null, windows.PAGE_READONLY, 0, 0, null) orelse return error.Unexpected;
    defer windows.CloseHandle(mapping);

    const view = MapViewOfFile(mapping, file_map_read, 0, 0, file_len) orelse return error.Unexpected;
    const mapped: []align(std.heap.page_size_min) const u8 = @as([*]align(std.heap.page_size_min) const u8, @ptrCast(@alignCast(view)))[0..file_len];
    return .{ .bytes = mapped };
}

const file_map_read: windows.DWORD = 0x0004;

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
