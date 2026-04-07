const std = @import("std");
const decoder_types = @import("../../decoder_types.zig");
const config = @import("config.zig");

pub const architecture = decoder_types.Architecture.bert;
pub const model_type = "bert";

pub const inspect_sample_tensors = [_][]const u8{
    "bert.embeddings.word_embeddings.weight",
    "bert.embeddings.position_embeddings.weight",
    "bert.encoder.layer.0.attention.self.query.weight",
    "bert.encoder.layer.0.attention.self.key.weight",
    "bert.encoder.layer.0.intermediate.dense.weight",
    "cls.predictions.transform.dense.weight",
};

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

pub fn supportsGeneration() bool {
    return false;
}

test "bert family loads parsed config into shared decoder config" {
    const testing = std.testing;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json =
        \\{
        \\  "model_type": "bert",
        \\  "hidden_size": 768,
        \\  "intermediate_size": 3072,
        \\  "layer_norm_eps": 1e-12,
        \\  "max_position_embeddings": 512,
        \\  "num_attention_heads": 12,
        \\  "num_hidden_layers": 12,
        \\  "vocab_size": 30522
        \\}
    ;
    var file = try tmp.dir.createFile("config.json", .{});
    defer file.close();
    try file.writeAll(config_json);

    const path = try tmp.dir.realpathAlloc(testing.allocator, "config.json");
    defer testing.allocator.free(path);

    var parsed = try loadParsedConfig(testing.allocator, path);
    defer parsed.deinit();

    try testing.expectEqual(decoder_types.Architecture.bert, parsed.value.architecture);
    try testing.expectEqualStrings("bert", parsed.value.model_type);
}
