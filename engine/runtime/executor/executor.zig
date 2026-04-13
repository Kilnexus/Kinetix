const std = @import("std");
const handle_mod = @import("../model/handle.zig");
const types = @import("../types.zig");

pub const Executor = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Executor {
        return .{ .allocator = allocator };
    }

    pub fn execute(self: Executor, handle: *const handle_mod.ModelHandle, plan: *const types.ExecutionPlan) !types.RuntimeResult {
        _ = self;
        _ = handle;
        _ = plan;
        return error.RuntimeExecutionNotImplemented;
    }
};
