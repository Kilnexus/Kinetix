const std = @import("std");
const chat_types = @import("chat_types.zig");
const decoder_types = @import("decoder_types.zig");
const legacy = @import("../../../legacy/zinfer/src/model/runtime/decoder_family.zig");

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;
pub const Tokenizer = legacy.Tokenizer;
pub const ParsedConfig = decoder_types.ParsedConfig;

pub const argMaxLogit = legacy.argMaxLogit;

pub fn detectArchitecture(model_type: []const u8) ?Architecture {
    return if (legacy.detectArchitecture(model_type)) |architecture|
        mapArchitectureFromLegacy(architecture)
    else
        null;
}

pub fn loadConfigFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    const parsed = try legacy.loadConfigFromFile(backing_allocator, path);
    return .{
        .arena = parsed.arena,
        .value = mapDecoderConfig(parsed.value),
    };
}

pub fn loadTokenizerFromModelDir(
    backing_allocator: std.mem.Allocator,
    architecture: Architecture,
    model_dir: []const u8,
) !Tokenizer {
    return try legacy.loadTokenizerFromModelDir(
        backing_allocator,
        mapArchitectureToLegacy(architecture),
        model_dir,
    );
}

pub fn supportsGeneration(architecture: Architecture) bool {
    return legacy.supportsGeneration(mapArchitectureToLegacy(architecture));
}

pub fn effectiveStopSequencesAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    extra_stop_sequences: [][]const u8,
) ![][]const u8 {
    return try legacy.effectiveStopSequencesAlloc(
        allocator,
        mapArchitectureToLegacy(architecture),
        extra_stop_sequences,
    );
}

pub fn isEosToken(architecture: Architecture, token_id: usize) bool {
    return legacy.isEosToken(mapArchitectureToLegacy(architecture), token_id);
}

pub fn architectureFromLegacy(value: legacy.Architecture) Architecture {
    return mapArchitectureFromLegacy(value);
}

pub fn architectureToLegacy(value: Architecture) legacy.Architecture {
    return mapArchitectureToLegacy(value);
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    messages: []const Message,
    mode: ThinkingMode,
) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const legacy_messages = try scratch.alloc(legacy.Message, messages.len);
    for (messages, legacy_messages) |message, *legacy_message| {
        const legacy_tool_calls = try scratch.alloc(legacy.ToolCall, message.tool_calls.len);
        for (message.tool_calls, legacy_tool_calls) |tool_call, *legacy_tool_call| {
            legacy_tool_call.* = .{
                .name = tool_call.name,
                .arguments_json = tool_call.arguments_json,
            };
        }

        legacy_message.* = .{
            .role = mapRoleToLegacy(message.role),
            .content = message.content,
            .tool_calls = legacy_tool_calls,
        };
    }

    return try legacy.renderMessagesPromptAlloc(
        allocator,
        mapArchitectureToLegacy(architecture),
        legacy_messages,
        mapThinkingModeToLegacy(mode),
    );
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    architecture: Architecture,
    user_text: []const u8,
    mode: ThinkingMode,
) ![]u8 {
    return try legacy.renderSingleUserPromptAlloc(
        allocator,
        mapArchitectureToLegacy(architecture),
        user_text,
        mapThinkingModeToLegacy(mode),
    );
}

fn mapArchitectureFromLegacy(value: legacy.Architecture) Architecture {
    return switch (value) {
        .qwen3 => .qwen3,
        .bert => .bert,
    };
}

fn mapArchitectureToLegacy(value: Architecture) legacy.Architecture {
    return switch (value) {
        .qwen3 => .qwen3,
        .bert => .bert,
    };
}

fn mapThinkingModeToLegacy(value: ThinkingMode) legacy.ThinkingMode {
    return switch (value) {
        .enabled => .enabled,
        .disabled => .disabled,
    };
}

fn mapRoleToLegacy(value: Role) legacy.Role {
    return switch (value) {
        .system => .system,
        .user => .user,
        .assistant => .assistant,
        .tool => .tool,
    };
}

fn mapDecoderConfig(value: legacy.DecoderConfig) decoder_types.DecoderConfig {
    return .{
        .architecture = mapArchitectureFromLegacy(value.architecture),
        .model_type = value.model_type,
        .hidden_size = value.hidden_size,
        .intermediate_size = value.intermediate_size,
        .num_hidden_layers = value.num_hidden_layers,
        .num_attention_heads = value.num_attention_heads,
        .num_key_value_heads = value.num_key_value_heads,
        .head_dim = value.head_dim,
        .vocab_size = value.vocab_size,
        .max_position_embeddings = value.max_position_embeddings,
        .rope_theta = value.rope_theta,
        .rms_norm_eps = value.rms_norm_eps,
        .torch_dtype = value.torch_dtype,
        .tie_word_embeddings = value.tie_word_embeddings,
    };
}

test "bridge exposes decoder family generation capability" {
    try std.testing.expect(supportsGeneration(.qwen3));
    try std.testing.expect(!supportsGeneration(.bert));
}
