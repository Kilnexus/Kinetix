const model = @import("tools/model.zig");
const text = @import("tools/text.zig");

pub const quantizeModelDir = model.quantizeModelDir;

pub const tokenizeText = text.tokenizeText;
pub const decodeIds = text.decodeIds;
