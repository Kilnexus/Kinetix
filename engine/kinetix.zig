pub const core = struct {
    pub const task = @import("core/task.zig");
};

pub const adapter = @import("adapter/adapter.zig");
pub const registry = @import("registry/registry.zig");
pub const scheduler = @import("scheduler/scheduler.zig");

test {
    _ = @import("testing/tests.zig");
}
