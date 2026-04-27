const types = @import("../../../types.zig");

pub const key = types.ProviderKey.paddleocr_ocr;
pub const modality = types.Modality.ocr;
pub const family_name = "paddleocr";

pub const backend = @import("backend.zig").backend;
pub const postprocess = @import("postprocess/index.zig");
pub const tryNormalize = @import("resolver.zig").tryNormalize;
