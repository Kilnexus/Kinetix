const std = @import("std");

const io = std.Options.debug_io;

pub const TtsSummary = struct {
    allocator: ?std.mem.Allocator = null,
    has_model_info: bool = false,
    has_session_options: bool = false,
    files: TtsFiles = .{},
    model_config: TtsModelConfig = .{},
    onnx: TtsOnnxContract = .{},

    pub fn deinit(self: *TtsSummary) void {
        const allocator = self.allocator orelse return;
        self.files.deinit(allocator);
        self.onnx.deinit(allocator);
        self.* = .{};
    }
};

pub const CodecConfig = struct {
    allocator: ?std.mem.Allocator = null,
    sample_rate: usize = 0,
    channels: usize = 0,
    num_quantizers: usize = 0,
    downsample_rate: usize = 0,
    files: CodecFiles = .{},
    onnx: CodecOnnxContract = .{},
    streaming_transformer_offset_count: usize = 0,
    streaming_attention_cache_count: usize = 0,

    pub fn deinit(self: *CodecConfig) void {
        const allocator = self.allocator orelse return;
        self.files.deinit(allocator);
        self.onnx.deinit(allocator);
        self.* = .{};
    }
};

pub const TtsFiles = struct {
    prefill: []const u8 = "",
    decode_step: []const u8 = "",
    local_decoder: []const u8 = "",
    local_cached_step: []const u8 = "",
    local_greedy_frame: []const u8 = "",
    local_fixed_sampled_frame: []const u8 = "",

    fn deinit(self: *TtsFiles, allocator: std.mem.Allocator) void {
        freeString(allocator, self.prefill);
        freeString(allocator, self.decode_step);
        freeString(allocator, self.local_decoder);
        freeString(allocator, self.local_cached_step);
        freeString(allocator, self.local_greedy_frame);
        freeString(allocator, self.local_fixed_sampled_frame);
        self.* = .{};
    }
};

pub const CodecFiles = struct {
    encode: []const u8 = "",
    decode_full: []const u8 = "",
    decode_step: []const u8 = "",

    fn deinit(self: *CodecFiles, allocator: std.mem.Allocator) void {
        freeString(allocator, self.encode);
        freeString(allocator, self.decode_full);
        freeString(allocator, self.decode_step);
        self.* = .{};
    }
};

pub const TtsModelConfig = struct {
    n_vq: usize = 0,
    row_width: usize = 0,
    hidden_size: usize = 0,
    global_layers: usize = 0,
    local_layers: usize = 0,
    vocab_size: usize = 0,
};

pub const TtsOnnxContract = struct {
    prefill_input_names: []const []const u8 = &.{},
    prefill_output_names: []const []const u8 = &.{},
    decode_input_names: []const []const u8 = &.{},
    decode_output_names: []const []const u8 = &.{},
    local_decoder_input_names: []const []const u8 = &.{},
    local_decoder_output_names: []const []const u8 = &.{},
    local_cached_input_names: []const []const u8 = &.{},
    local_cached_output_names: []const []const u8 = &.{},
    local_fixed_sampled_frame_input_names: []const []const u8 = &.{},
    local_fixed_sampled_frame_output_names: []const []const u8 = &.{},

    fn deinit(self: *TtsOnnxContract, allocator: std.mem.Allocator) void {
        freeStringList(allocator, self.prefill_input_names);
        freeStringList(allocator, self.prefill_output_names);
        freeStringList(allocator, self.decode_input_names);
        freeStringList(allocator, self.decode_output_names);
        freeStringList(allocator, self.local_decoder_input_names);
        freeStringList(allocator, self.local_decoder_output_names);
        freeStringList(allocator, self.local_cached_input_names);
        freeStringList(allocator, self.local_cached_output_names);
        freeStringList(allocator, self.local_fixed_sampled_frame_input_names);
        freeStringList(allocator, self.local_fixed_sampled_frame_output_names);
        self.* = .{};
    }
};

pub const CodecOnnxContract = struct {
    encode_input_names: []const []const u8 = &.{},
    encode_output_names: []const []const u8 = &.{},
    decode_input_names: []const []const u8 = &.{},
    decode_output_names: []const []const u8 = &.{},
    decode_step_input_names: []const []const u8 = &.{},
    decode_step_output_names: []const []const u8 = &.{},

    fn deinit(self: *CodecOnnxContract, allocator: std.mem.Allocator) void {
        freeStringList(allocator, self.encode_input_names);
        freeStringList(allocator, self.encode_output_names);
        freeStringList(allocator, self.decode_input_names);
        freeStringList(allocator, self.decode_output_names);
        freeStringList(allocator, self.decode_step_input_names);
        freeStringList(allocator, self.decode_step_output_names);
        self.* = .{};
    }
};

pub fn loadTtsSummary(allocator: std.mem.Allocator, path: []const u8) !TtsSummary {
    const value = try loadJsonValue(allocator, path);
    defer value.deinit();

    if (value.value != .object) return error.InvalidMossTtsMeta;
    const object = value.value.object;
    var summary = TtsSummary{
        .allocator = allocator,
        .has_model_info = object.get("model_info") != null,
        .has_session_options = object.get("session_options") != null,
        .files = try parseTtsFiles(allocator, object.get("files")),
        .model_config = parseTtsModelConfig(object.get("model_config")),
        .onnx = try parseTtsOnnxContract(allocator, object.get("onnx")),
    };
    errdefer summary.deinit();
    return summary;
}

pub fn loadCodecConfig(allocator: std.mem.Allocator, path: []const u8) !CodecConfig {
    const value = try loadJsonValue(allocator, path);
    defer value.deinit();

    if (value.value != .object) return error.InvalidMossCodecMeta;
    const object = value.value.object;
    var config = parseCodecConfigObject(object.get("codec_config"));
    config.allocator = allocator;
    config.files = try parseCodecFiles(allocator, object.get("files"));
    config.onnx = try parseCodecOnnxContract(allocator, object.get("onnx"));
    config.streaming_transformer_offset_count = countObjectArrayField(object.get("streaming_decode"), "transformer_offsets");
    config.streaming_attention_cache_count = countObjectArrayField(object.get("streaming_decode"), "attention_caches");
    errdefer config.deinit();
    return config;
}

fn loadJsonValue(allocator: std.mem.Allocator, path: []const u8) !std.json.Parsed(std.json.Value) {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);
    return try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
}

fn parseTtsFiles(allocator: std.mem.Allocator, raw: ?std.json.Value) !TtsFiles {
    const object = objectValue(raw) orelse return .{};
    return .{
        .prefill = try copyStringField(allocator, object, "prefill"),
        .decode_step = try copyStringField(allocator, object, "decode_step"),
        .local_decoder = try copyStringField(allocator, object, "local_decoder"),
        .local_cached_step = try copyStringField(allocator, object, "local_cached_step"),
        .local_greedy_frame = try copyStringField(allocator, object, "local_greedy_frame"),
        .local_fixed_sampled_frame = try copyStringField(allocator, object, "local_fixed_sampled_frame"),
    };
}

fn parseCodecFiles(allocator: std.mem.Allocator, raw: ?std.json.Value) !CodecFiles {
    const object = objectValue(raw) orelse return .{};
    return .{
        .encode = try copyStringField(allocator, object, "encode"),
        .decode_full = try copyStringField(allocator, object, "decode_full"),
        .decode_step = try copyStringField(allocator, object, "decode_step"),
    };
}

fn parseTtsModelConfig(raw: ?std.json.Value) TtsModelConfig {
    const object = objectValue(raw) orelse return .{};
    return .{
        .n_vq = unsignedField(object, "n_vq", 0),
        .row_width = unsignedField(object, "row_width", 0),
        .hidden_size = unsignedField(object, "hidden_size", 0),
        .global_layers = unsignedField(object, "global_layers", 0),
        .local_layers = unsignedField(object, "local_layers", 0),
        .vocab_size = unsignedField(object, "vocab_size", 0),
    };
}

fn parseTtsOnnxContract(allocator: std.mem.Allocator, raw: ?std.json.Value) !TtsOnnxContract {
    const object = objectValue(raw) orelse return .{};
    return .{
        .prefill_input_names = try copyFixedStringList(allocator, &.{ "input_ids", "attention_mask" }),
        .prefill_output_names = try copyStringListField(allocator, object, "prefill_output_names"),
        .decode_input_names = try copyStringListField(allocator, object, "decode_input_names"),
        .decode_output_names = try copyStringListField(allocator, object, "decode_output_names"),
        .local_decoder_input_names = try copyFixedStringList(allocator, &.{ "global_hidden", "text_token_id", "audio_prefix_token_ids" }),
        .local_decoder_output_names = try copyFixedStringList(allocator, &.{ "text_logits", "audio_logits" }),
        .local_cached_input_names = try copyStringListField(allocator, object, "local_cached_input_names"),
        .local_cached_output_names = try copyStringListField(allocator, object, "local_cached_output_names"),
        .local_fixed_sampled_frame_input_names = try copyStringListField(allocator, object, "local_fixed_sampled_frame_input_names"),
        .local_fixed_sampled_frame_output_names = try copyStringListField(allocator, object, "local_fixed_sampled_frame_output_names"),
    };
}

fn parseCodecConfigObject(raw: ?std.json.Value) CodecConfig {
    const object = objectValue(raw) orelse return .{};
    return .{
        .sample_rate = unsignedField(object, "sample_rate", 0),
        .channels = unsignedField(object, "channels", 0),
        .num_quantizers = unsignedField(object, "num_quantizers", 0),
        .downsample_rate = unsignedField(object, "downsample_rate", 0),
    };
}

fn parseCodecOnnxContract(allocator: std.mem.Allocator, raw: ?std.json.Value) !CodecOnnxContract {
    const object = objectValue(raw) orelse return .{};
    return .{
        .encode_input_names = try copyStringListField(allocator, object, "encode_input_names"),
        .encode_output_names = try copyStringListField(allocator, object, "encode_output_names"),
        .decode_input_names = try copyStringListField(allocator, object, "decode_input_names"),
        .decode_output_names = try copyStringListField(allocator, object, "decode_output_names"),
        .decode_step_input_names = try copyStringListField(allocator, object, "decode_step_input_names"),
        .decode_step_output_names = try copyStringListField(allocator, object, "decode_step_output_names"),
    };
}

fn objectValue(raw: ?std.json.Value) ?std.json.ObjectMap {
    const value = raw orelse return null;
    return switch (value) {
        .object => |object| object,
        else => null,
    };
}

fn countObjectArrayField(raw: ?std.json.Value, field_name: []const u8) usize {
    const object = objectValue(raw) orelse return 0;
    const value = object.get(field_name) orelse return 0;
    return switch (value) {
        .array => |items| items.items.len,
        else => 0,
    };
}

fn copyStringField(allocator: std.mem.Allocator, object: std.json.ObjectMap, field_name: []const u8) ![]const u8 {
    const value = object.get(field_name) orelse return "";
    return switch (value) {
        .string => |item| try allocator.dupe(u8, item),
        else => "",
    };
}

fn copyStringListField(allocator: std.mem.Allocator, object: std.json.ObjectMap, field_name: []const u8) ![]const []const u8 {
    const value = object.get(field_name) orelse return &.{};
    if (value != .array) return &.{};
    const out = try allocator.alloc([]const u8, value.array.items.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |item| allocator.free(item);
        allocator.free(out);
    }
    for (value.array.items, out) |item, *slot| {
        slot.* = switch (item) {
            .string => |text| try allocator.dupe(u8, text),
            else => try allocator.dupe(u8, ""),
        };
        filled += 1;
    }
    return out;
}

fn copyFixedStringList(allocator: std.mem.Allocator, items: []const []const u8) ![]const []const u8 {
    const out = try allocator.alloc([]const u8, items.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |item| allocator.free(item);
        allocator.free(out);
    }
    for (items, out) |item, *slot| {
        slot.* = try allocator.dupe(u8, item);
        filled += 1;
    }
    return out;
}

fn unsignedField(object: std.json.ObjectMap, field_name: []const u8, default: usize) usize {
    const value = object.get(field_name) orelse return default;
    return switch (value) {
        .integer => |item| if (item < 0) default else @intCast(item),
        else => default,
    };
}

fn freeString(allocator: std.mem.Allocator, value: []const u8) void {
    if (value.len != 0) allocator.free(value);
}

fn freeStringList(allocator: std.mem.Allocator, items: []const []const u8) void {
    for (items) |item| freeString(allocator, item);
    allocator.free(items);
}

test "moss tts codec meta parser extracts audio contract" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "codec_browser_onnx_meta.json",
        \\{
        \\  "codec_config": {
        \\    "sample_rate": 48000,
        \\    "channels": 2,
        \\    "num_quantizers": 32
        \\  }
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "codec_browser_onnx_meta.json");
    defer std.testing.allocator.free(path);

    var config = try loadCodecConfig(std.testing.allocator, path);
    defer config.deinit();
    try std.testing.expectEqual(@as(usize, 48000), config.sample_rate);
    try std.testing.expectEqual(@as(usize, 2), config.channels);
    try std.testing.expectEqual(@as(usize, 32), config.num_quantizers);
}

test "moss tts meta parser extracts graph contract" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "tts_browser_onnx_meta.json",
        \\{
        \\  "model_info": {},
        \\  "session_options": {},
        \\  "files": {
        \\    "prefill": "moss_tts_prefill.onnx",
        \\    "decode_step": "moss_tts_decode_step.onnx",
        \\    "local_decoder": "moss_tts_local_decoder.onnx",
        \\    "local_cached_step": "moss_tts_local_cached_step.onnx",
        \\    "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx"
        \\  },
        \\  "model_config": {
        \\    "n_vq": 16,
        \\    "row_width": 17,
        \\    "hidden_size": 768,
        \\    "global_layers": 12,
        \\    "local_layers": 1,
        \\    "vocab_size": 16384
        \\  },
        \\  "onnx": {
        \\    "prefill_output_names": ["global_hidden", "present_key_0"],
        \\    "decode_input_names": ["input_ids", "past_valid_lengths", "past_key_0"],
        \\    "decode_output_names": ["global_hidden", "present_key_0"],
        \\    "local_cached_input_names": ["global_hidden", "text_token_id"],
        \\    "local_cached_output_names": ["text_logits", "audio_logits"],
        \\    "local_fixed_sampled_frame_input_names": ["global_hidden", "repetition_seen_mask"],
        \\    "local_fixed_sampled_frame_output_names": ["should_continue", "frame_token_ids"]
        \\  }
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "tts_browser_onnx_meta.json");
    defer std.testing.allocator.free(path);

    var summary = try loadTtsSummary(std.testing.allocator, path);
    defer summary.deinit();

    try std.testing.expect(summary.has_model_info);
    try std.testing.expect(summary.has_session_options);
    try std.testing.expectEqualStrings("moss_tts_prefill.onnx", summary.files.prefill);
    try std.testing.expectEqualStrings("moss_tts_local_cached_step.onnx", summary.files.local_cached_step);
    try std.testing.expectEqual(@as(usize, 16), summary.model_config.n_vq);
    try std.testing.expectEqual(@as(usize, 2), summary.onnx.prefill_input_names.len);
    try std.testing.expectEqualStrings("input_ids", summary.onnx.prefill_input_names[0]);
    try std.testing.expectEqual(@as(usize, 3), summary.onnx.decode_input_names.len);
    try std.testing.expectEqual(@as(usize, 2), summary.onnx.local_cached_output_names.len);
}

test "moss codec meta parser extracts graph and streaming decode contract" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "codec_browser_onnx_meta.json",
        \\{
        \\  "files": {
        \\    "encode": "moss_audio_tokenizer_encode.onnx",
        \\    "decode_full": "moss_audio_tokenizer_decode_full.onnx",
        \\    "decode_step": "moss_audio_tokenizer_decode_step.onnx"
        \\  },
        \\  "codec_config": {
        \\    "sample_rate": 48000,
        \\    "channels": 2,
        \\    "downsample_rate": 3840,
        \\    "num_quantizers": 16
        \\  },
        \\  "onnx": {
        \\    "encode_input_names": ["waveform", "input_lengths"],
        \\    "encode_output_names": ["audio_codes", "audio_code_lengths"],
        \\    "decode_input_names": ["audio_codes", "audio_code_lengths"],
        \\    "decode_output_names": ["audio", "audio_lengths"],
        \\    "decode_step_input_names": ["audio_codes", "audio_code_lengths", "transformer_offset_0"],
        \\    "decode_step_output_names": ["audio", "audio_lengths", "transformer_offset_out_0"]
        \\  },
        \\  "streaming_decode": {
        \\    "transformer_offsets": [{}, {}],
        \\    "attention_caches": [{}, {}, {}]
        \\  }
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "codec_browser_onnx_meta.json");
    defer std.testing.allocator.free(path);

    var config = try loadCodecConfig(std.testing.allocator, path);
    defer config.deinit();

    try std.testing.expectEqualStrings("moss_audio_tokenizer_decode_step.onnx", config.files.decode_step);
    try std.testing.expectEqual(@as(usize, 3840), config.downsample_rate);
    try std.testing.expectEqual(@as(usize, 3), config.onnx.decode_step_input_names.len);
    try std.testing.expectEqual(@as(usize, 2), config.streaming_transformer_offset_count);
    try std.testing.expectEqual(@as(usize, 3), config.streaming_attention_cache_count);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
