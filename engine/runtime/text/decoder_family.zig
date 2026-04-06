const std = @import("std");
const legacy = @import("../../../legacy/zinfer/src/model/runtime/decoder_family.zig");

pub const Architecture = legacy.Architecture;
pub const ThinkingMode = legacy.ThinkingMode;
pub const Role = legacy.Role;
pub const Message = legacy.Message;
pub const Tokenizer = legacy.Tokenizer;
pub const ParsedConfig = legacy.ParsedConfig;

pub const detectArchitecture = legacy.detectArchitecture;
pub const loadConfigFromFile = legacy.loadConfigFromFile;
pub const loadTokenizerFromModelDir = legacy.loadTokenizerFromModelDir;
pub const supportsGeneration = legacy.supportsGeneration;
pub const isEosToken = legacy.isEosToken;
pub const effectiveStopSequencesAlloc = legacy.effectiveStopSequencesAlloc;
pub const renderMessagesPromptAlloc = legacy.renderMessagesPromptAlloc;
pub const renderSingleUserPromptAlloc = legacy.renderSingleUserPromptAlloc;
pub const argMaxLogit = legacy.argMaxLogit;

test "bridge exposes decoder family generation capability" {
    try std.testing.expect(supportsGeneration(.qwen3));
    try std.testing.expect(!supportsGeneration(.bert));
}
