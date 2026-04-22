const std = @import("std");
const fs_compat = @import("engine_fs_compat");

pub fn readModelTypeAlloc(allocator: std.mem.Allocator, config_path: []const u8) !?[]u8 {
    const bytes = try fs_compat.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();

    const model_type_value = parsed.value.object.get("model_type") orelse return null;
    if (model_type_value != .string) return error.InvalidModelType;
    return try allocator.dupe(u8, model_type_value.string);
}
