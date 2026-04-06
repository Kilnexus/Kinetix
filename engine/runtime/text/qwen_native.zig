const std = @import("std");
const backend = @import("../../artifacts/backend/backend.zig");
const task = @import("../../core/task.zig");
const text_prompts = @import("prompts.zig");
const text_options = @import("generate_options.zig");
const text_runtime = @import("generator_runtime.zig");
const zinfer_decoder_family = @import("../../../legacy/zinfer/src/model/runtime/decoder_family.zig");
const zinfer_kv_cache = @import("../../../legacy/zinfer/src/model/runtime/optimized_kv_cache.zig");
const zinfer_optimized_decoder = @import("../../../legacy/zinfer/src/model/runtime/optimized_decoder.zig");
const zinfer_tensor_backend = @import("../../../legacy/zinfer/src/tensor/backends/backend.zig");

pub const NativeBatchOutput = struct {
    texts: [][]u8,
    total_decoded_tokens: usize,
    finished_requests: usize,

    pub fn deinit(self: *NativeBatchOutput, allocator: std.mem.Allocator) void {
        for (self.texts) |text| allocator.free(text);
        allocator.free(self.texts);
        self.* = undefined;
    }
};

pub fn generateSingleUserText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    user_text: []const u8,
    options: text_options.GenerateOptions,
) ![]u8 {
    var runtime = try text_runtime.GeneratorRuntime.init(
        allocator,
        model_dir,
        options.backend_scheme,
        options.thread_count,
    );
    defer runtime.deinit();

    const prompt = try text_prompts.buildSingleUserPromptAlloc(
        allocator,
        runtime.model.cfg.architecture,
        user_text,
        options.system_prompt,
        options.thinking_mode,
    );
    defer allocator.free(prompt);

    return try runtime.generateFromPrompt(prompt, options);
}

pub fn executeQwenSingle(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    request: task.TaskRequest,
) ![]u8 {
    const input = switch (request.input) {
        .text => |value| value,
        .none => "",
        else => return error.InvalidInputPayload,
    };

    const options = text_options.GenerateOptions{
        .max_new_tokens = request.generation.max_tokens orelse 64,
        .thinking_mode = .disabled,
        .system_prompt = null,
        .sampling = text_options.defaultSamplingConfig(.disabled),
        .seed = 0,
        .stream_output = false,
        .stop_sequences = &.{},
        .backend_scheme = mapBackendScheme(preferred_weights),
        .kv_cache_scheme = .auto,
        .q8_layout = zinfer_kv_cache.default_q8_layout,
        .thread_count = 0,
    };

    return try generateSingleUserText(allocator, model_dir, input, options);
}

pub fn executeQwenBatch(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme,
    requests: []const task.TaskRequest,
) !NativeBatchOutput {
    if (requests.len == 0) return error.InvalidBatchSize;

    const resolved_max_tokens = resolveMaxTokens(requests);
    if (resolvedMaxTokensMismatch(requests, resolved_max_tokens)) return error.InconsistentBatchGenerationOptions;

    var runtime = try text_runtime.GeneratorRuntime.init(
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

        prompt.* = try text_prompts.buildSingleUserPromptAlloc(
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

    const generated_tokens = try allocator.alloc(std.ArrayListUnmanaged(u32), requests.len);
    defer {
        for (generated_tokens) |*tokens| tokens.deinit(allocator);
        allocator.free(generated_tokens);
    }
    for (generated_tokens) |*tokens| tokens.* = .empty;

    var active_requests = requests.len;
    var finished_requests: usize = 0;
    var total_decoded_tokens: usize = 0;

    for (0..resolved_max_tokens) |_| {
        if (active_requests == 0) break;

        var progressed = false;
        for (batch.requests, generated_tokens) |*request_state, *tokens| {
            if (!request_state.active) continue;

            const next_token = try zinfer_decoder_family.argMaxLogit(request_state.workspace.logits);
            if (zinfer_decoder_family.isEosToken(batch.model.cfg.architecture, next_token)) {
                request_state.active = false;
                active_requests -= 1;
                finished_requests += 1;
                continue;
            }

            try tokens.append(allocator, std.math.cast(u32, next_token) orelse return error.TokenIdOutOfRange);
            _ = try batch.model.forwardTokenId(&request_state.workspace, &request_state.cache, next_token);
            request_state.decoded_tokens += 1;
            total_decoded_tokens += 1;
            progressed = true;
        }

        if (!progressed) break;
    }

    const texts = try allocator.alloc([]u8, requests.len);
    errdefer {
        for (texts, 0..) |text, index| {
            if (index >= requests.len) break;
            allocator.free(text);
        }
        allocator.free(texts);
    }

    for (generated_tokens, texts) |tokens, *text| {
        if (tokens.items.len == 0) {
            text.* = try allocator.dupe(u8, "");
            continue;
        }
        text.* = try runtime.tokenizer.decodeAlloc(allocator, tokens.items);
    }

    return .{
        .texts = texts,
        .total_decoded_tokens = total_decoded_tokens,
        .finished_requests = finished_requests,
    };
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
