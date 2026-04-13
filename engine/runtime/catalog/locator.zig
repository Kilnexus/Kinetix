const std = @import("std");

pub fn duplicatePath(allocator: std.mem.Allocator, value: ?[]const u8) !?[]u8 {
    if (value) |path| return try allocator.dupe(u8, path);
    return null;
}
