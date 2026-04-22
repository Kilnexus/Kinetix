const types = @import("../../../types.zig");

pub const key = types.ProviderKey.qwen3_text;
pub const modality = types.Modality.text;
pub const family_name = "qwen3";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;
