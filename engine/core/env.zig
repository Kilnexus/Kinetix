const builtin = @import("builtin");
const std = @import("std");

pub fn getOwned(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi or builtin.os.tag == .emscripten or builtin.os.tag == .freestanding or builtin.os.tag == .other) {
        const environ: std.process.Environ = .{ .block = .global };
        return environ.getAlloc(allocator, name);
    }
    if (!builtin.link_libc) return error.EnvironmentVariableMissing;

    const c_name = try allocator.dupeZ(u8, name);
    defer allocator.free(c_name);
    const value = std.c.getenv(c_name.ptr) orelse return error.EnvironmentVariableMissing;
    return allocator.dupe(u8, std.mem.span(value));
}
