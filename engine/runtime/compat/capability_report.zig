const std = @import("std");
const types = @import("../types.zig");

pub const CompatibilityReport = struct {
    allocator: std.mem.Allocator,
    status: types.CompatibilityStatus,
    warnings: []types.CompatibilityWarning,
    rewrites: []types.CompatibilityRewrite,

    pub fn init(
        allocator: std.mem.Allocator,
        status: types.CompatibilityStatus,
        warnings: []const types.CompatibilityWarning,
        rewrites: []const types.CompatibilityRewrite,
    ) !CompatibilityReport {
        const owned_warnings = try allocator.alloc(types.CompatibilityWarning, warnings.len);
        errdefer allocator.free(owned_warnings);
        const owned_rewrites = try allocator.alloc(types.CompatibilityRewrite, rewrites.len);
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

    pub fn supported(allocator: std.mem.Allocator) !CompatibilityReport {
        return try init(allocator, .supported, &.{}, &.{});
    }

    pub fn deinit(self: *CompatibilityReport) void {
        self.allocator.free(self.warnings);
        self.allocator.free(self.rewrites);
        self.* = undefined;
    }
};
