const std = @import("std");
const decoder_types = @import("../../../../../../engine/runtime/text/decoder_types.zig");
const chat_types = @import("../../../../../../engine/runtime/text/chat_types.zig");
const generic_block = @import("../../layers/rmsnorm_gqa_swiglu_block.zig");
const weights_layout = @import("../../layers/weights_layout.zig");
const wordpiece_tokenizer = @import("../../../tokenizer/wordpiece.zig");
const config = @import("config.zig");

pub const architecture = decoder_types.Architecture.bert;
pub const model_type = "bert";

pub const TokenizerImpl = wordpiece_tokenizer.Tokenizer;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Message = chat_types.Message;

pub const eos_token_ids = &[_]u32{};
pub const default_stop_sequences = &[_][]const u8{};
pub const common_weights = weights_layout.CommonWeights{
    .embed_tokens_weight = "",
    .final_norm_weight = "",
    .lm_head_weight = "",
};
pub const layer_layout = generic_block.LayerLayout{};

pub fn loadParsedConfig(backing_allocator: std.mem.Allocator, path: []const u8) !decoder_types.ParsedConfig {
    var parsed = try config.loadFromFile(backing_allocator, path);
    errdefer parsed.deinit();

    return .{
        .arena = parsed.arena,
        .value = .{
            .architecture = architecture,
            .model_type = parsed.value.model_type,
            .hidden_size = parsed.value.hidden_size,
            .intermediate_size = parsed.value.intermediate_size,
            .num_hidden_layers = parsed.value.num_hidden_layers,
            .num_attention_heads = parsed.value.num_attention_heads,
            .num_key_value_heads = parsed.value.num_attention_heads,
            .head_dim = parsed.value.hidden_size / parsed.value.num_attention_heads,
            .vocab_size = parsed.value.vocab_size,
            .max_position_embeddings = parsed.value.max_position_embeddings,
            .rope_theta = 0.0,
            .rms_norm_eps = parsed.value.layer_norm_eps,
            .torch_dtype = parsed.value.torch_dtype,
            .tie_word_embeddings = false,
        },
    };
}

pub fn loadTokenizerFromModelDir(backing_allocator: std.mem.Allocator, model_dir: []const u8) !TokenizerImpl {
    return try wordpiece_tokenizer.Tokenizer.loadFromModelDir(backing_allocator, model_dir);
}

pub fn layerTensorNameAlloc(
    allocator: std.mem.Allocator,
    _: usize,
    _: weights_layout.LayerTensorKind,
) ![]u8 {
    _ = allocator;
    return error.UnsupportedArchitectureForDecoderRuntime;
}

pub fn renderMessagesPromptAlloc(
    allocator: std.mem.Allocator,
    _: []const Message,
    _: ThinkingMode,
) ![]u8 {
    _ = allocator;
    return error.UnsupportedArchitectureForGeneration;
}

pub fn renderSingleUserPromptAlloc(
    allocator: std.mem.Allocator,
    _: []const u8,
    _: ThinkingMode,
) ![]u8 {
    _ = allocator;
    return error.UnsupportedArchitectureForGeneration;
}

pub fn assistantHistoryContent(content: []const u8) []const u8 {
    return content;
}

test "adapter family loads bert config into shared decoder config" {
    const testing = std.testing;

    var parsed = try loadParsedConfig(testing.allocator, "models/bert-base-uncased/config.json");
    defer parsed.deinit();

    try testing.expectEqual(decoder_types.Architecture.bert, parsed.value.architecture);
    try testing.expectEqualStrings("bert", parsed.value.model_type);
    try testing.expectEqual(@as(usize, 768), parsed.value.hidden_size);
    try testing.expectEqual(@as(usize, 64), parsed.value.head_dim);
}
