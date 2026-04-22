const types = @import("../../../types.zig");

pub const key = types.ProviderKey.chandra_ocr;
pub const modality = types.Modality.ocr;
pub const family_name = "chandra";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;

pub const native = @import("native.zig");
pub const preprocess = @import("preprocess.zig");
pub const store = @import("store.zig");
pub const vision = @import("vision.zig");
pub const weights = @import("weights.zig");
