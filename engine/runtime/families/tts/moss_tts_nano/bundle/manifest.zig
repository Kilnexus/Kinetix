const std = @import("std");

const io = std.Options.debug_io;

pub const Summary = struct {
    allocator: ?std.mem.Allocator = null,
    builtin_voice_count: usize = 0,
    has_generation_defaults: bool = false,
    has_model_files: bool = false,
    tts_config: TtsConfig = .{},
    prompt_templates: PromptTemplates = .{},
    generation_defaults: GenerationDefaults = .{},
    default_voice_name: []const u8 = &.{},
    default_prompt_audio_codes: []const []const i32 = &.{},

    pub fn deinit(self: *Summary) void {
        const allocator = self.allocator orelse return;
        allocator.free(self.prompt_templates.user_prompt_prefix_token_ids);
        allocator.free(self.prompt_templates.user_prompt_after_reference_token_ids);
        allocator.free(self.prompt_templates.assistant_prompt_prefix_token_ids);
        for (self.default_prompt_audio_codes) |row| allocator.free(row);
        allocator.free(self.default_prompt_audio_codes);
        allocator.free(self.default_voice_name);
        self.* = .{};
    }
};

pub const TtsConfig = struct {
    n_vq: usize = 0,
    audio_pad_token_id: i32 = 0,
    pad_token_id: i32 = 0,
    im_start_token_id: i32 = 0,
    im_end_token_id: i32 = 0,
    audio_start_token_id: i32 = 0,
    audio_end_token_id: i32 = 0,
    audio_user_slot_token_id: i32 = 0,
    audio_assistant_slot_token_id: i32 = 0,
    vocab_size: usize = 0,
};

pub const PromptTemplates = struct {
    user_prompt_prefix_token_ids: []const i32 = &.{},
    user_prompt_after_reference_token_ids: []const i32 = &.{},
    assistant_prompt_prefix_token_ids: []const i32 = &.{},
};

pub const GenerationDefaults = struct {
    max_new_frames: usize = 0,
    do_sample: bool = false,
    sample_mode: []const u8 = "",
};

pub fn loadSummary(allocator: std.mem.Allocator, path: []const u8) !Summary {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidMossTtsManifest;
    const object = parsed.value.object;

    var summary = Summary{
        .allocator = allocator,
        .builtin_voice_count = countArrayField(object, "builtin_voices"),
        .has_generation_defaults = object.get("generation_defaults") != null,
        .has_model_files = object.get("model_files") != null,
        .tts_config = parseTtsConfig(object.get("tts_config")),
        .prompt_templates = try parsePromptTemplates(allocator, object.get("prompt_templates")),
        .generation_defaults = parseGenerationDefaults(object.get("generation_defaults")),
        .default_voice_name = try copyDefaultVoiceName(allocator, object.get("builtin_voices")),
        .default_prompt_audio_codes = try copyDefaultPromptAudioCodes(allocator, object.get("builtin_voices")),
    };
    errdefer summary.deinit();
    return summary;
}

fn countArrayField(object: std.json.ObjectMap, field_name: []const u8) usize {
    const value = object.get(field_name) orelse return 0;
    return switch (value) {
        .array => |items| items.items.len,
        else => 0,
    };
}

fn parseTtsConfig(raw: ?std.json.Value) TtsConfig {
    const value = raw orelse return .{};
    if (value != .object) return .{};
    const object = value.object;
    return .{
        .n_vq = unsignedField(object, "n_vq", 0),
        .audio_pad_token_id = intField(object, "audio_pad_token_id", 0),
        .pad_token_id = intField(object, "pad_token_id", 0),
        .im_start_token_id = intField(object, "im_start_token_id", 0),
        .im_end_token_id = intField(object, "im_end_token_id", 0),
        .audio_start_token_id = intField(object, "audio_start_token_id", 0),
        .audio_end_token_id = intField(object, "audio_end_token_id", 0),
        .audio_user_slot_token_id = intField(object, "audio_user_slot_token_id", 0),
        .audio_assistant_slot_token_id = intField(object, "audio_assistant_slot_token_id", 0),
        .vocab_size = unsignedField(object, "vocab_size", 0),
    };
}

fn parsePromptTemplates(allocator: std.mem.Allocator, raw: ?std.json.Value) !PromptTemplates {
    const value = raw orelse return .{};
    if (value != .object) return .{};
    const object = value.object;
    return .{
        .user_prompt_prefix_token_ids = try copyIntArray(allocator, object.get("user_prompt_prefix_token_ids")),
        .user_prompt_after_reference_token_ids = try copyIntArray(allocator, object.get("user_prompt_after_reference_token_ids")),
        .assistant_prompt_prefix_token_ids = try copyIntArray(allocator, object.get("assistant_prompt_prefix_token_ids")),
    };
}

fn parseGenerationDefaults(raw: ?std.json.Value) GenerationDefaults {
    const value = raw orelse return .{};
    if (value != .object) return .{};
    const object = value.object;
    return .{
        .max_new_frames = unsignedField(object, "max_new_frames", 0),
        .do_sample = boolField(object, "do_sample", false),
        .sample_mode = stringField(object, "sample_mode", ""),
    };
}

fn copyDefaultVoiceName(allocator: std.mem.Allocator, raw: ?std.json.Value) ![]const u8 {
    const object = firstVoiceObject(raw) orelse return &.{};
    const voice = object.get("voice") orelse return &.{};
    if (voice != .string) return &.{};
    return try allocator.dupe(u8, voice.string);
}

fn copyDefaultPromptAudioCodes(allocator: std.mem.Allocator, raw: ?std.json.Value) ![]const []const i32 {
    const object = firstVoiceObject(raw) orelse return &.{};
    return try copyIntMatrix(allocator, object.get("prompt_audio_codes"));
}

fn firstVoiceObject(raw: ?std.json.Value) ?std.json.ObjectMap {
    const value = raw orelse return null;
    if (value != .array) return null;
    if (value.array.items.len == 0) return null;
    const first = value.array.items[0];
    if (first != .object) return null;
    return first.object;
}

fn copyIntMatrix(allocator: std.mem.Allocator, raw: ?std.json.Value) ![]const []const i32 {
    const value = raw orelse return &.{};
    if (value != .array) return &.{};
    const rows = try allocator.alloc([]const i32, value.array.items.len);
    var filled: usize = 0;
    errdefer {
        for (rows[0..filled]) |row| allocator.free(row);
        allocator.free(rows);
    }
    for (value.array.items, rows) |row_value, *row| {
        row.* = try copyIntArray(allocator, row_value);
        filled += 1;
    }
    return rows;
}

fn copyIntArray(allocator: std.mem.Allocator, raw: ?std.json.Value) ![]const i32 {
    const value = raw orelse return &.{};
    if (value != .array) return &.{};
    const out = try allocator.alloc(i32, value.array.items.len);
    errdefer allocator.free(out);
    for (value.array.items, out) |item, *slot| {
        slot.* = intValue(item, 0);
    }
    return out;
}

fn intField(object: std.json.ObjectMap, field_name: []const u8, default: i32) i32 {
    return intValue(object.get(field_name) orelse return default, default);
}

fn unsignedField(object: std.json.ObjectMap, field_name: []const u8, default: usize) usize {
    const value = intField(object, field_name, @intCast(default));
    if (value < 0) return default;
    return @intCast(value);
}

fn boolField(object: std.json.ObjectMap, field_name: []const u8, default: bool) bool {
    const value = object.get(field_name) orelse return default;
    return switch (value) {
        .bool => |item| item,
        else => default,
    };
}

fn stringField(object: std.json.ObjectMap, field_name: []const u8, default: []const u8) []const u8 {
    const value = object.get(field_name) orelse return default;
    return switch (value) {
        .string => |item| item,
        else => default,
    };
}

fn intValue(value: std.json.Value, default: i32) i32 {
    return switch (value) {
        .integer => |item| @intCast(item),
        else => default,
    };
}

test "moss tts manifest summary counts builtin voices" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "browser_poc_manifest.json",
        \\{
        \\  "builtin_voices": ["a", "b"],
        \\  "generation_defaults": {},
        \\  "model_files": {}
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "browser_poc_manifest.json");
    defer std.testing.allocator.free(path);

    const summary = try loadSummary(std.testing.allocator, path);
    defer @constCast(&summary).deinit();
    try std.testing.expectEqual(@as(usize, 2), summary.builtin_voice_count);
    try std.testing.expect(summary.has_generation_defaults);
    try std.testing.expect(summary.has_model_files);
}

test "moss tts manifest summary extracts request contract" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "browser_poc_manifest.json",
        \\{
        \\  "tts_config": {
        \\    "n_vq": 2,
        \\    "audio_pad_token_id": 1024,
        \\    "audio_start_token_id": 6,
        \\    "audio_end_token_id": 7,
        \\    "audio_user_slot_token_id": 8,
        \\    "audio_assistant_slot_token_id": 9,
        \\    "vocab_size": 16384
        \\  },
        \\  "prompt_templates": {
        \\    "user_prompt_prefix_token_ids": [10, 11],
        \\    "user_prompt_after_reference_token_ids": [12],
        \\    "assistant_prompt_prefix_token_ids": [13, 14]
        \\  },
        \\  "generation_defaults": {
        \\    "max_new_frames": 375,
        \\    "do_sample": true,
        \\    "sample_mode": "fixed"
        \\  },
        \\  "builtin_voices": [
        \\    {
        \\      "voice": "Junhao",
        \\      "prompt_audio_codes": [[1, 2], [3, 4]]
        \\    }
        \\  ],
        \\  "model_files": {}
        \\}
    );

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "browser_poc_manifest.json");
    defer std.testing.allocator.free(path);

    var summary = try loadSummary(std.testing.allocator, path);
    defer summary.deinit();

    try std.testing.expectEqual(@as(usize, 2), summary.tts_config.n_vq);
    try std.testing.expectEqual(@as(i32, 1024), summary.tts_config.audio_pad_token_id);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11 }, summary.prompt_templates.user_prompt_prefix_token_ids);
    try std.testing.expectEqualStrings("Junhao", summary.default_voice_name);
    try std.testing.expectEqual(@as(usize, 2), summary.default_prompt_audio_codes.len);
    try std.testing.expectEqualSlices(i32, &.{ 3, 4 }, summary.default_prompt_audio_codes[1]);
    try std.testing.expectEqual(@as(usize, 375), summary.generation_defaults.max_new_frames);
    try std.testing.expect(summary.generation_defaults.do_sample);
    try std.testing.expectEqualStrings("fixed", summary.generation_defaults.sample_mode);
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
