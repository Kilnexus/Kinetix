const types = @import("../../types.zig");

pub const key = types.ProviderKey.generic;
pub const modality = types.Modality.multimodal;
pub const family_name = "generic";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;
