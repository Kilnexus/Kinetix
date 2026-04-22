const types = @import("../../../types.zig");

pub const key = types.ProviderKey.swiftocr_ocr;
pub const modality = types.Modality.ocr;
pub const family_name = "swiftocr";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;
pub const native = @import("native.zig");
