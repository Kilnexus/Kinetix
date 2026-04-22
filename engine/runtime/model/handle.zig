const std = @import("std");
const backend_mod = @import("../backend/backend.zig");
const normalized = @import("resolver/normalized_model.zig");

pub const ModelHandle = struct {
    allocator: std.mem.Allocator,
    normalized: normalized.NormalizedModel,
    runtime_backend: *const backend_mod.RuntimeBackend,
    provider_state: ?*anyopaque = null,

    pub fn deinit(self: *ModelHandle) void {
        self.runtime_backend.deinitState(self.allocator, self.provider_state);
        self.normalized.deinit();
        self.* = undefined;
    }
};
