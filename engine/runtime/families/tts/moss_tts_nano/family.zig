const types = @import("../../../types.zig");

pub const key = types.ProviderKey.moss_tts_nano_tts;
pub const modality = types.Modality.tts;
pub const family_name = "moss_tts_nano";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;

pub const bundle = struct {
    pub const locator = @import("bundle/locator.zig");
};
