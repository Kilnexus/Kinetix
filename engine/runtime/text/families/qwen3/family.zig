const std = @import("std");
const fs_compat = @import("engine_fs_compat");
const decoder_types = @import("../../decoder_types.zig");
const chat_types = @import("../common/chat_types.zig");
const bpe_tokenizer = @import("../../bpe.zig");
const config = @import("config.zig");
const generation_policy = @import("generation_policy.zig");
const layout = @import("layout.zig");
const chat_template = @import("chat_template.zig");
const weights = @import("weights.zig");

const ChandraTextConfig = struct {
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: ?f64 = null,
    rope_parameters: ?std.json.Value = null,
    rms_norm_eps: ?f64 = null,
    torch_dtype: ?[]const u8 = null,
    tie_word_embeddings: ?bool = null,
};

const ChandraConfig = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,
    text_config: ChandraTextConfig,
};

pub const architecture = decoder_types.Architecture.qwen3;
pub const model_type = "qwen3";

pub const TokenizerImpl = bpe_tokenizer.Tokenizer;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;

pub const eos_token_ids = generation_policy.eos_token_ids;
pub const default_stop_sequences = generation_policy.default_stop_sequences;
pub const common_weights = weights.common_weights;
pub const layer_layout = layout.layer_layout;
pub const layerTensorNameAlloc = weights.layerTensorNameAlloc;
pub const renderMessagesPromptAlloc = chat_template.renderMessagesPromptAlloc;
pub const renderSingleUserPromptAlloc = chat_template.renderSingleUserPromptAlloc;
pub const assistantHistoryContent = chat_template.assistantHistoryContent;
pub const inspect_sample_tensors = [_][]const u8{
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
};

const ResolvedRope = struct {
    theta: f64,
    position_mode: decoder_types.RopePositionMode = .scalar,
    mrope_sections: [4]u32 = .{ 0, 0, 0, 0 },
};

pub fn loadParsedConfig(backing_allocator: std.mem.Allocator, path: []const u8) !decoder_types.ParsedConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try fs_compat.cwd().readFileAlloc(allocator, path, 2 * 1024 * 1024);

    if (std.json.parseFromSliceLeaky(config.Config, allocator, bytes, .{
        .ignore_unknown_fields = true,
    })) |parsed| {
        return .{
            .arena = arena,
            .value = .{
                .architecture = architecture,
                .model_type = parsed.model_type,
                .hidden_size = parsed.hidden_size,
                .intermediate_size = parsed.intermediate_size,
                .num_hidden_layers = parsed.num_hidden_layers,
                .num_attention_heads = parsed.num_attention_heads,
                .num_key_value_heads = parsed.num_key_value_heads,
                .head_dim = parsed.head_dim,
                .vocab_size = parsed.vocab_size,
                .max_position_embeddings = parsed.max_position_embeddings,
                .rope_theta = parsed.rope_theta,
                .rope_position_mode = .scalar,
                .mrope_sections = .{ 0, 0, 0, 0 },
                .rms_norm_eps = parsed.rms_norm_eps,
                .torch_dtype = parsed.torch_dtype,
                .tie_word_embeddings = parsed.tie_word_embeddings,
            },
        };
    } else |_| {}

    const chandra = try std.json.parseFromSliceLeaky(ChandraConfig, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
    const rope = resolveChandraRope(chandra.text_config);
    return .{
        .arena = arena,
        .value = .{
            .architecture = architecture,
            .model_type = chandra.text_config.model_type,
            .hidden_size = chandra.text_config.hidden_size,
            .intermediate_size = chandra.text_config.intermediate_size,
            .num_hidden_layers = chandra.text_config.num_hidden_layers,
            .num_attention_heads = chandra.text_config.num_attention_heads,
            .num_key_value_heads = chandra.text_config.num_key_value_heads,
            .head_dim = chandra.text_config.head_dim,
            .vocab_size = chandra.text_config.vocab_size,
            .max_position_embeddings = chandra.text_config.max_position_embeddings,
            .rope_theta = rope.theta,
            .rope_position_mode = rope.position_mode,
            .mrope_sections = rope.mrope_sections,
            .rms_norm_eps = chandra.text_config.rms_norm_eps orelse 1e-6,
            .torch_dtype = chandra.text_config.torch_dtype orelse "bfloat16",
            .tie_word_embeddings = chandra.text_config.tie_word_embeddings orelse false,
        },
    };
}

fn resolveChandraRope(text_config: ChandraTextConfig) ResolvedRope {
    var resolved = ResolvedRope{
        .theta = text_config.rope_theta orelse 1_000_000.0,
    };
    const rope_parameters = text_config.rope_parameters orelse return resolved;
    if (rope_parameters != .object) return resolved;

    if (rope_parameters.object.get("full_attention")) |full_attention| {
        if (full_attention == .object) {
            if (full_attention.object.get("rope_theta")) |theta_value| {
                if (jsonNumberToF64(theta_value)) |theta| resolved.theta = theta;
            }
        }
    } else if (rope_parameters.object.get("rope_theta")) |theta_value| {
        if (jsonNumberToF64(theta_value)) |theta| resolved.theta = theta;
    }

    if (rope_parameters.object.get("mrope_section")) |sections_value| {
        if (sections_value == .array) {
            resolved.position_mode = .mrope;
            for (0..@min(sections_value.array.items.len, resolved.mrope_sections.len)) |index| {
                const item = sections_value.array.items[index];
                resolved.mrope_sections[index] = jsonNumberToU32(item) orelse 0;
            }
        }
    }

    return resolved;
}

fn jsonNumberToF64(value: std.json.Value) ?f64 {
    return switch (value) {
        .integer => |number| @floatFromInt(number),
        .float => |number| number,
        else => null,
    };
}

fn jsonNumberToU32(value: std.json.Value) ?u32 {
    return switch (value) {
        .integer => |number| std.math.cast(u32, number),
        .float => |number| blk: {
            if (number < 0 or @floor(number) != number) break :blk null;
            break :blk std.math.cast(u32, @as(i64, @intFromFloat(number)));
        },
        else => null,
    };
}

pub fn loadTokenizerFromModelDir(backing_allocator: std.mem.Allocator, model_dir: []const u8) !TokenizerImpl {
    return try bpe_tokenizer.Tokenizer.loadFromModelDir(backing_allocator, model_dir);
}

pub fn supportsGeneration() bool {
    return true;
}

test "adapter family loads parsed config into shared decoder config" {
    const testing = std.testing;

    var parsed = try loadParsedConfig(testing.allocator, "models/text/Qwen3-0.6B/config.json");
    defer parsed.deinit();

    try testing.expectEqual(decoder_types.Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3", parsed.value.model_type);
}

test "adapter family loads chandra wrapped qwen text config into shared decoder config" {
    const testing = std.testing;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "config.json",
        .data =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 160,
        \\    "vocab_size": 248320,
        \\    "max_position_embeddings": 262144
        \\  }
        \\}
        ,
    });

    const path = try tmp.dir.realpathAlloc(testing.allocator, "config.json");
    defer testing.allocator.free(path);

    var parsed = try loadParsedConfig(testing.allocator, path);
    defer parsed.deinit();

    try testing.expectEqual(decoder_types.Architecture.qwen3, parsed.value.architecture);
    try testing.expectEqualStrings("qwen3_5_text", parsed.value.model_type);
    try testing.expectEqual(@as(usize, 32), parsed.value.num_hidden_layers);
    try testing.expectEqual(@as(f64, 1_000_000.0), parsed.value.rope_theta);
    try testing.expectEqual(@as(f64, 1e-6), parsed.value.rms_norm_eps);
    try testing.expectEqualStrings("bfloat16", parsed.value.torch_dtype);
    try testing.expect(!parsed.value.tie_word_embeddings);
    try testing.expectEqual(decoder_types.RopePositionMode.scalar, parsed.value.rope_position_mode);
    try testing.expectEqualSlices(u32, &.{ 0, 0, 0, 0 }, &parsed.value.mrope_sections);
}

test "adapter family resolves chandra rope parameters into shared decoder config" {
    const testing = std.testing;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{
        .sub_path = "config.json",
        .data =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {
        \\    "model_type": "qwen3_5_text",
        \\    "hidden_size": 2560,
        \\    "intermediate_size": 9216,
        \\    "num_hidden_layers": 32,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 4,
        \\    "head_dim": 160,
        \\    "vocab_size": 248320,
        \\    "max_position_embeddings": 262144,
        \\    "rope_parameters": {
        \\      "full_attention": {
        \\        "rope_theta": 250000.0
        \\      },
        \\      "mrope_section": [16, 24, 24]
        \\    }
        \\  }
        \\}
        ,
    });

    const path = try tmp.dir.realpathAlloc(testing.allocator, "config.json");
    defer testing.allocator.free(path);

    var parsed = try loadParsedConfig(testing.allocator, path);
    defer parsed.deinit();

    try testing.expectEqual(@as(f64, 250000.0), parsed.value.rope_theta);
    try testing.expectEqual(decoder_types.RopePositionMode.mrope, parsed.value.rope_position_mode);
    try testing.expectEqualSlices(u32, &.{ 16, 24, 24, 0 }, &parsed.value.mrope_sections);
}
