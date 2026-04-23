const types = @import("../../../types.zig");

pub const key = types.ProviderKey.moss_tts_nano_tts;
pub const modality = types.Modality.tts;
pub const family_name = "moss_tts_nano";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;

pub const bundle = struct {
    pub const api = @import("bundle/index.zig");
    pub const paths = @import("bundle/paths.zig");
    pub const manifest = @import("bundle/manifest.zig");
    pub const meta = @import("bundle/meta.zig");
};
