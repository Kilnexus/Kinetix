pub const io = struct {
    pub const image = @import("io/image.zig");
    pub const preprocess = @import("io/preprocess.zig");
};
pub const analysis = struct {
    pub const inspect = @import("analysis/inspect.zig");
};
pub const memory = struct {
    pub const reuse_allocator = @import("memory/reuse_allocator.zig");
};
pub const api = struct {
    pub const base = @import("base.zig");
    pub const engine = @import("engine.zig");
    pub const runtime = @import("runtime.zig");
};
pub const image = io.image;
pub const preprocess = io.preprocess;
pub const inspect = analysis.inspect;
pub const base = api.base;
pub const engine = api.engine;
pub const runtime = api.runtime;
pub const modules = struct {
    pub const blocks = @import("modules/blocks.zig");
    pub const detect = @import("modules/detect.zig");
    pub const psa = @import("modules/psa.zig");
};
pub const blocks = modules.blocks;
pub const detect = modules.detect;
pub const psa = modules.psa;
