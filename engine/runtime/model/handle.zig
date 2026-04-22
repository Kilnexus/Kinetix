const std = @import("std");
const normalized = @import("resolver/normalized_model.zig");

pub const ModelHandle = struct {
    allocator: std.mem.Allocator,
    normalized: normalized.NormalizedModel,
    provider_state: ?*anyopaque = null,

    pub fn deinit(self: *ModelHandle) void {
        self.normalized.deinit();
        self.* = undefined;
    }
};
