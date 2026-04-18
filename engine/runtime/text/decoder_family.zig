const std = @import("std");
const decoder_types = @import("decoder_types.zig");
const logits_util = @import("logits.zig");
const family_registry = @import("families/registry.zig");

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = family_registry.ThinkingMode;
pub const Role = family_registry.Role;
pub const ToolCall = family_registry.ToolCall;
pub const Message = family_registry.Message;
pub const TopLogit = logits_util.TopLogit;
pub const CommonWeights = family_registry.CommonWeights;
pub const LayerTensorKind = family_registry.LayerTensorKind;
pub const DecoderConfig = decoder_types.DecoderConfig;
pub const ParsedConfig = decoder_types.ParsedConfig;
pub const Tokenizer = family_registry.Tokenizer;
pub const RopePositionMode = decoder_types.RopePositionMode;
pub const TokenPosition = decoder_types.TokenPosition;

pub const argMaxLogit = logits_util.argMaxLogit;
pub const topKLogitsAlloc = logits_util.topKLogitsAlloc;

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    return family_registry.detectArchitecture(model_type);
}

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    return try family_registry.loadParsedConfig(backing_allocator, path);
}

pub fn loadTokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    architecture: Architecture,
    model_dir: []const u8,
) !Tokenizer {
    return try family_registry.loadTokenizerFromModelDir(backing_allocator, architecture, model_dir);
}

pub fn eosTokenIds(architecture: Architecture) []const u32 {
    return family_registry.eosTokenIds(architecture);
}

pub fn isEosToken(architecture: Architecture, token_id: usize) bool {
    for (eosTokenIds(architecture)) |eos_id| {
        if (token_id == eos_id) return true;
    }
    return false;
}

pub fn defaultStopSequences(architecture: Architecture) []const []const u8 {
    return family_registry.defaultStopSequences(architecture);
}

pub fn effectiveStopSequencesAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    extra_stop_sequences: [][]const u8,
) ![][]const u8 {
    const defaults = defaultStopSequences(architecture);
    var unique_extra_count: usize = 0;

    for (extra_stop_sequences, 0..) |stop_sequence, idx| {
        if (containsStopSequence(defaults, stop_sequence)) continue;
        if (containsStopSequence(extra_stop_sequences[0..idx], stop_sequence)) continue;
        unique_extra_count += 1;
    }

    const combined = try allocator.alloc([]const u8, defaults.len + unique_extra_count);
    var count: usize = 0;

    for (defaults) |stop_sequence| {
        combined[count] = stop_sequence;
        count += 1;
    }

    for (extra_stop_sequences, 0..) |stop_sequence, idx| {
        if (containsStopSequence(defaults, stop_sequence)) continue;
        if (containsStopSequence(extra_stop_sequences[0..idx], stop_sequence)) continue;
        combined[count] = stop_sequence;
        count += 1;
    }

    return combined;
}

pub fn commonWeights(architecture: Architecture) CommonWeights {
    return family_registry.commonWeights(architecture);
}

pub fn layerLayout(architecture: Architecture) @TypeOf(family_registry.layerLayout(.qwen3)) {
    return family_registry.layerLayout(architecture);
}

pub fn layerTensorNameAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    layer_index: usize,
    kind: LayerTensorKind,
) ![]u8 {
    return try family_registry.layerTensorNameAlloc(allocator, architecture, layer_index, kind);
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return try family_registry.renderMessagesPromptAlloc(allocator, architecture, messages, mode);
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return try family_registry.renderSingleUserPromptAlloc(allocator, architecture, user_text, mode);
}

pub fn assistantHistoryContent(
    architecture: Architecture,
    content: []const u8,
) []const u8 {
    return family_registry.assistantHistoryContent(architecture, content);
}

pub fn inspectSampleTensorNames(architecture: Architecture) []const []const u8 {
    return family_registry.inspectSampleTensorNames(architecture);
}

pub fn supportsGeneration(architecture: Architecture) bool {
    return family_registry.supportsGeneration(architecture);
}

fn containsStopSequence(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |existing| {
        if (std.mem.eql(u8, existing, needle)) return true;
    }
    return false;
}

test "family detects qwen3 model type" {
    const testing = std.testing;
    try testing.expectEqual(Architecture.qwen3, detectArchitecture("qwen3").?);
    try testing.expectEqual(Architecture.bert, detectArchitecture("bert").?);
    try testing.expect(detectArchitecture("unknown-model") == null);
}

test "family loads qwen3 config through shared registry" {
    const testing = std.testing;

    var parsed = try loadConfigFromFile(testing.allocator, "models/text/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}

test "family exposes qwen3 generation capability" {
    try std.testing.expect(supportsGeneration(.qwen3));
    try std.testing.expect(!supportsGeneration(.bert));
}
