const std = @import("std");
const chat_types = @import("chat_types.zig");
const decoder_types = @import("decoder_types.zig");
const generic_block = @import("block_layout.zig");
const logits_util = @import("logits.zig");
const weights_layout = @import("weights_layout.zig");
const bert_family = @import("families/bert/family.zig");
const qwen3_family = @import("families/qwen3/family.zig");

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;
pub const TopLogit = logits_util.TopLogit;
pub const CommonWeights = weights_layout.CommonWeights;
pub const LayerTensorKind = weights_layout.LayerTensorKind;
pub const DecoderConfig = decoder_types.DecoderConfig;
pub const ParsedConfig = decoder_types.ParsedConfig;

pub const Tokenizer = union(Architecture) {
    qwen3: qwen3_family.TokenizerImpl,
    bert: void,

    pub fn loadFromModelDir(
        backing_allocator: std.mem.Allocator,
        architecture: Architecture,
        model_dir: []const u8,
    ) !Tokenizer {
        return switch (architecture) {
            .qwen3 => .{
                .qwen3 = try qwen3_family.loadTokenizerFromModelDir(backing_allocator, model_dir),
            },
            .bert => error.UnsupportedArchitectureForGeneration,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            .qwen3 => |*tokenizer| tokenizer.deinit(),
            .bert => {},
        }
    }

    pub fn encodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        return switch (self.*) {
            .qwen3 => |*tokenizer| tokenizer.encodeAlloc(allocator, text),
            .bert => error.UnsupportedArchitectureForGeneration,
        };
    }

    pub fn decodeAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        return switch (self.*) {
            .qwen3 => |*tokenizer| tokenizer.decodeAlloc(allocator, ids),
            .bert => error.UnsupportedArchitectureForGeneration,
        };
    }
};

pub const argMaxLogit = logits_util.argMaxLogit;
pub const topKLogitsAlloc = logits_util.topKLogitsAlloc;

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    if (std.mem.eql(u8, model_type, qwen3_family.model_type)) return .qwen3;
    if (std.mem.eql(u8, model_type, bert_family.model_type)) return .bert;
    return null;
}

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    const architecture = try detectArchitectureFromConfigFile(backing_allocator, path);
    return switch (architecture) {
        .qwen3 => try qwen3_family.loadParsedConfig(backing_allocator, path),
        .bert => try bert_family.loadParsedConfig(backing_allocator, path),
    };
}

pub fn loadTokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    architecture: Architecture,
    model_dir: []const u8,
) !Tokenizer {
    return try Tokenizer.loadFromModelDir(backing_allocator, architecture, model_dir);
}

pub fn eosTokenIds(architecture: Architecture) []const u32 {
    return switch (architecture) {
        .qwen3 => qwen3_family.eos_token_ids,
        .bert => &.{},
    };
}

pub fn isEosToken(architecture: Architecture, token_id: usize) bool {
    for (eosTokenIds(architecture)) |eos_id| {
        if (token_id == eos_id) return true;
    }
    return false;
}

pub fn defaultStopSequences(architecture: Architecture) []const []const u8 {
    return switch (architecture) {
        .qwen3 => qwen3_family.default_stop_sequences,
        .bert => &.{},
    };
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
    return switch (architecture) {
        .qwen3 => qwen3_family.common_weights,
        .bert => .{
            .embed_tokens_weight = "",
            .final_norm_weight = "",
            .lm_head_weight = "",
        },
    };
}

pub fn layerLayout(architecture: Architecture) generic_block.LayerLayout {
    return switch (architecture) {
        .qwen3 => qwen3_family.layer_layout,
        .bert => .{},
    };
}

pub fn layerTensorNameAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    layer_index: usize,
    kind: LayerTensorKind,
) ![]u8 {
    return switch (architecture) {
        .qwen3 => try qwen3_family.layerTensorNameAlloc(allocator, layer_index, kind),
        .bert => error.UnsupportedArchitectureForDecoderRuntime,
    };
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    return switch (architecture) {
        .qwen3 => try qwen3_family.renderMessagesPromptAlloc(allocator, messages, mode),
        .bert => error.UnsupportedArchitectureForGeneration,
    };
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return switch (architecture) {
        .qwen3 => try qwen3_family.renderSingleUserPromptAlloc(allocator, user_text, mode),
        .bert => error.UnsupportedArchitectureForGeneration,
    };
}

pub fn assistantHistoryContent(
    architecture: Architecture,
    content: []const u8,
) []const u8 {
    return switch (architecture) {
        .qwen3 => qwen3_family.assistantHistoryContent(content),
        .bert => content,
    };
}

pub fn inspectSampleTensorNames(architecture: Architecture) []const []const u8 {
    return switch (architecture) {
        .qwen3 => &qwen3_family.inspect_sample_tensors,
        .bert => &bert_family.inspect_sample_tensors,
    };
}

pub fn supportsGeneration(architecture: Architecture) bool {
    return switch (architecture) {
        .qwen3 => qwen3_family.supportsGeneration(),
        .bert => bert_family.supportsGeneration(),
    };
}

fn containsStopSequence(haystack: []const []const u8, needle: []const u8) bool {
    for (haystack) |existing| {
        if (std.mem.eql(u8, existing, needle)) return true;
    }
    return false;
}

fn detectArchitectureFromConfigFile(
    backing_allocator: std.mem.Allocator,
    path: []const u8,
) !Architecture {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const bytes = try readFileAllocAtPath(allocator, path, 1024 * 1024);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    const model_type_value = parsed.value.object.get("model_type") orelse return error.MissingModelType;
    if (model_type_value != .string) return error.InvalidModelType;

    return detectArchitecture(model_type_value.string) orelse error.UnsupportedModelType;
}

fn readFileAllocAtPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        return file.readToEndAlloc(allocator, max_bytes);
    }
    return std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
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
