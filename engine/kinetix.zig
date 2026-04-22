pub const core = struct {
    pub const task = @import("core/task.zig");
    pub const memory = struct {
        pub const ReuseAllocator = @import("core/memory/reuse_allocator.zig").ReuseAllocator;
        pub const ArenaPool = @import("core/memory/arena_pool.zig").ArenaPool;
        pub const ReuseStats = @import("core/memory/reuse_allocator.zig").Stats;
    };
};
pub const artifacts = @import("artifacts/artifacts.zig");
pub const runtime = @import("runtime/runtime.zig");

test {
    _ = runtime;
}
