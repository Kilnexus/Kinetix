const std = @import("std");
const chat_types = @import("../chat_types.zig");
const decoder_types = @import("../decoder_types.zig");
const generic_block = @import("../block_layout.zig");
const weights_layout = @import("../weights_layout.zig");
const bert_family = @import("bert/family.zig");
const qwen3_family = @import("qwen3/family.zig");

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Message = chat_types.Message;
pub const ParsedConfig = decoder_types.ParsedConfig;
pub const CommonWeights = weights_layout.CommonWeights;
pub const LayerTensorKind = weights_layout.LayerTensorKind;

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    if (std.mem.eql(u8, model_type, qwen3_family.model_type)) return .qwen3;
    if (std.mem.eql(u8, model_type, bert_family.model_type)) return .bert;
    return null;
}

pub fn loadParsedConfig(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    const architecture = try detectArchitectureFromConfigFile(backing_allocator, path);
    return switch (architecture) {
        .qwen3 => try qwen3_family.loadParsedConfig(backing_allocator, path),
        .bert => try bert_family.loadParsedConfig(backing_allocator, path),
    };
}

pub fn eosTokenIds(architecture: Architecture) []const u32 {
    return switch (architecture) {
        .qwen3 => qwen3_family.eos_token_ids,
        .bert => &.{},
    };
}

pub fn defaultStopSequences(architecture: Architecture) []const []const u8 {
    return switch (architecture) {
        .qwen3 => qwen3_family.default_stop_sequences,
        .bert => &.{},
    };
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

test "registry detects known family architectures" {
    const testing = std.testing;
    try testing.expectEqual(Architecture.qwen3, detectArchitecture("qwen3").?);
    try testing.expectEqual(Architecture.bert, detectArchitecture("bert").?);
    try testing.expect(detectArchitecture("unknown-model") == null);
}
