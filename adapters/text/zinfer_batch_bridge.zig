const std = @import("std");
const backend = @import("../../engine/artifacts/backend/backend.zig");
const task = @import("../../engine/core/task.zig");
const zinfer_prompts = @import("../../legacy/zinfer/src/app/cli/prompts.zig");
const zinfer_runtime = @import("../../legacy/zinfer/src/app/cli/runtime.zig");
const zinfer_kv_cache = @import("../../legacy/zinfer/src/model/runtime/optimized_kv_cache.zig");
const zinfer_optimized_decoder = @import("../../legacy/zinfer/src/model/runtime/optimized_decoder.zig");
const zinfer_tensor_backend = @import("../../legacy/zinfer/src/tensor/backends/backend.zig");

pub fn executeQwenBatch(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    requests: []const task.TaskRequest,
) !void {
    if (requests.len == 0) return;

    const resolved_max_tokens = resolveMaxTokens(requests);
    if (resolvedMaxTokensMismatch(requests, resolved_max_tokens)) return error.InconsistentBatchGenerationOptions;

    var runtime = try zinfer_runtime.GeneratorRuntime.init(
        allocator,
        model_dir,
        mapBackendScheme(preferred_weights),
        0,
    );
    defer runtime.deinit();

    const resolved_kv_cache_scheme = zinfer_kv_cache.resolveScheme(.auto, runtime.model.backendName());

    const prompts = try allocator.alloc([]u8, requests.len);
    defer {
        for (prompts) |prompt| allocator.free(prompt);
        allocator.free(prompts);
    }

    const prompt_ids = try allocator.alloc([]usize, requests.len);
    defer {
        for (prompt_ids) |ids| allocator.free(ids);
        allocator.free(prompt_ids);
    }

    var max_prompt_len: usize = 0;
    for (requests, prompts, prompt_ids) |request, *prompt, *ids| {
        const text = switch (request.input) {
            .text => |value| value,
            .none => "",
            else => return error.InvalidInputPayload,
        };

        prompt.* = try zinfer_prompts.buildSingleUserPromptAlloc(
            allocator,
            runtime.model.cfg.architecture,
            text,
            null,
            .disabled,
        );

        const encoded_u32 = try runtime.tokenizer.encodeAlloc(allocator, prompt.*);
        defer allocator.free(encoded_u32);
        if (encoded_u32.len == 0) return error.EmptyPrompt;

        ids.* = try allocator.alloc(usize, encoded_u32.len);
        for (encoded_u32, 0..) |token_id, index| {
            ids.*[index] = token_id;
        }
        max_prompt_len = @max(max_prompt_len, ids.*.len);
    }

    var batch = try zinfer_optimized_decoder.BatchRuntime.init(
        allocator,
        &runtime.model,
        requests.len,
        max_prompt_len + resolved_max_tokens,
        resolved_kv_cache_scheme,
        zinfer_kv_cache.default_q8_layout,
    );
    defer batch.deinit();

    for (prompt_ids, 0..) |ids, request_index| {
        try batch.prefillPromptIds(request_index, ids);
    }

    _ = try batch.decodeRoundRobinArgMax(resolved_max_tokens);
}

fn resolveMaxTokens(requests: []const task.TaskRequest) usize {
    return requests[0].generation.max_tokens orelse 64;
}

fn resolvedMaxTokensMismatch(requests: []const task.TaskRequest, expected: usize) bool {
    for (requests) |request| {
        if ((request.generation.max_tokens orelse 64) != expected) return true;
    }
    return false;
}

fn mapBackendScheme(scheme: backend.WeightScheme) zinfer_tensor_backend.Scheme {
    return switch (scheme) {
        .auto => .auto,
        .bf16 => .bf16,
        .q8 => .q8,
        .q6 => .q6,
        .q4 => .q4,
    };
}
