const types = @import("../../../types.zig");

pub const key = types.ProviderKey.yolo_vision;
pub const modality = types.Modality.vision;
pub const family_name = "yolo";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;
