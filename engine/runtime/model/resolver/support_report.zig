const std = @import("std");
const types = @import("../../types.zig");

pub const RuntimeSupportReport = struct {
    allocator: std.mem.Allocator,
    status: types.RuntimeSupportStatus,
    warnings: []types.RuntimeSupportWarning,
    rewrites: []types.RuntimeSupportRewrite,

    pub fn init(
        allocator: std.mem.Allocator,
        status: types.RuntimeSupportStatus,
        warnings: []const types.RuntimeSupportWarning,
        rewrites: []const types.RuntimeSupportRewrite,
    ) !RuntimeSupportReport {
        const owned_warnings = try allocator.alloc(types.RuntimeSupportWarning, warnings.len);
        errdefer allocator.free(owned_warnings);
        const owned_rewrites = try allocator.alloc(types.RuntimeSupportRewrite, rewrites.len);
        errdefer allocator.free(owned_rewrites);

        @memcpy(owned_warnings, warnings);
        @memcpy(owned_rewrites, rewrites);

        return .{
            .allocator = allocator,
            .status = status,
            .warnings = owned_warnings,
            .rewrites = owned_rewrites,
        };
    }

    pub fn supported(allocator: std.mem.Allocator) !RuntimeSupportReport {
        return try init(allocator, .supported, &.{}, &.{});
    }

    pub fn deinit(self: *RuntimeSupportReport) void {
        self.allocator.free(self.warnings);
        self.allocator.free(self.rewrites);
        self.* = undefined;
    }
};
