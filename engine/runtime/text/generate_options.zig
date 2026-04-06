const std = @import("std");
const optimized_kv_cache = @import("../../../legacy/zinfer/src/model/runtime/optimized_kv_cache.zig");
const decoder_family = @import("../../../legacy/zinfer/src/model/runtime/decoder_family.zig");
const tensor_backend = @import("../../../legacy/zinfer/src/tensor/backends/backend.zig");
const sampler = @import("../../../legacy/zinfer/src/sampling/sampler.zig");

pub const GenerateOptions = struct {
    max_new_tokens: usize,
    thinking_mode: decoder_family.ThinkingMode,
    system_prompt: ?[]const u8,
    sampling: sampler.SamplingConfig,
    seed: u64,
    stream_output: bool,
    stop_sequences: [][]const u8,
    backend_scheme: tensor_backend.Scheme,
    kv_cache_scheme: optimized_kv_cache.Scheme,
    q8_layout: optimized_kv_cache.Q8Layout,
    thread_count: usize,

    pub fn deinit(self: *GenerateOptions, allocator: std.mem.Allocator) void {
        allocator.free(self.stop_sequences);
    }
};

pub fn defaultSamplingConfig(mode: decoder_family.ThinkingMode) sampler.SamplingConfig {
    return switch (mode) {
        .enabled => .{
            .temperature = 0.6,
            .top_k = 20,
            .top_p = 0.95,
            .min_p = 0.0,
            .presence_penalty = 0.0,
            .frequency_penalty = 0.0,
            .repetition_penalty = 1.1,
        },
        .disabled => .{
            .temperature = 0.7,
            .top_k = 20,
            .top_p = 0.8,
            .min_p = 0.0,
            .presence_penalty = 0.0,
            .frequency_penalty = 0.0,
            .repetition_penalty = 1.1,
        },
    };
}
