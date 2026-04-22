const types = @import("../../../types.zig");

pub const key = types.ProviderKey.bert_text;
pub const modality = types.Modality.text;
pub const family_name = "bert";

pub const backend = @import("backend.zig").backend;
pub const tryNormalize = @import("resolver.zig").tryNormalize;
