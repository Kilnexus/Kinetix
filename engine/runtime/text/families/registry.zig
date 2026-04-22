const std = @import("std");
const chat_types = @import("common/chat_types.zig");
const bpe_tokenizer = @import("../bpe.zig");
const decoder_types = @import("../decoder_types.zig");
const generic_block = @import("../block_layout.zig");
const weights_layout = @import("../weights_layout.zig");
const bert_family = @import("bert/family.zig");
const qwen3_family = @import("qwen3/family.zig");
const io = std.Options.debug_io;

pub const Architecture = decoder_types.Architecture;
pub const ThinkingMode = chat_types.ThinkingMode;
pub const Role = chat_types.Role;
pub const ToolCall = chat_types.ToolCall;
pub const Message = chat_types.Message;
pub const ParsedConfig = decoder_types.ParsedConfig;
pub const CommonWeights = weights_layout.CommonWeights;
pub const LayerTensorKind = weights_layout.LayerTensorKind;
pub const Tokenizer = union(Architecture) {
    qwen3: bpe_tokenizer.Tokenizer,
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

    if (detectArchitecture(model_type_value.string)) |architecture| return architecture;

    if (parsed.value.object.get("text_config")) |text_config_value| {
        if (text_config_value == .object) {
            if (text_config_value.object.get("model_type")) |text_model_type_value| {
                if (text_model_type_value != .string) return error.InvalidModelType;
                if (std.mem.eql(u8, text_model_type_value.string, "qwen3_5_text")) return .qwen3;
            }
        }
    }

    return error.UnsupportedModelType;
}

fn readFileAllocAtPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        var file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);
        var buffer: [4096]u8 = undefined;
        var reader = file.reader(io, &buffer);
        return reader.interface.allocRemaining(allocator, .limited(max_bytes));
    }
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes));
}

test "registry detects known family architectures" {
    const testing = std.testing;
    try testing.expectEqual(Architecture.qwen3, detectArchitecture("qwen3").?);
    try testing.expectEqual(Architecture.bert, detectArchitecture("bert").?);
    try testing.expect(detectArchitecture("unknown-model") == null);
}

test "registry detects qwen3 architecture from chandra wrapped config" {
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

    try testing.expectEqual(Architecture.qwen3, try detectArchitectureFromConfigFile(testing.allocator, path));
}
