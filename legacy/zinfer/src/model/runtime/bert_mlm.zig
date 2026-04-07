const std = @import("std");
const cpu = @import("../../kernel/core/cpu.zig");
const attention = @import("../../kernel/attention/basic.zig");
const logits_util = @import("../layers/logits.zig");
const bert_family = @import("../families/bert/family.zig");
const decoder_types = @import("../../../../../engine/runtime/text/decoder_types.zig");
const tensor_backend = @import("../../tensor/backends/backend.zig");
const parallel_rows = @import("../../tensor/parallel/parallel_rows.zig");

const word_embeddings_name = "bert.embeddings.word_embeddings.weight";
const position_embeddings_name = "bert.embeddings.position_embeddings.weight";
const mlm_transform_weight_name = "cls.predictions.transform.dense.weight";

pub const Prediction = struct {
    token_id: u32,
    token: []u8,
    logit: f32,
};

pub const FillMaskResult = struct {
    mask_position: usize,
    predictions: []Prediction,

    pub fn deinit(self: *FillMaskResult, allocator: std.mem.Allocator) void {
        for (self.predictions) |prediction| {
            allocator.free(prediction.token);
        }
        allocator.free(self.predictions);
    }
};

pub const EmbeddingMode = enum {
    cls,
    mean,

    pub fn name(self: EmbeddingMode) []const u8 {
        return switch (self) {
            .cls => "cls",
            .mean => "mean",
        };
    }
};

const EncoderOutput = struct {
    hidden: []f32,
    seq_len: usize,
};

const Workspace = struct {
    hidden_a: []f32,
    hidden_b: []f32,
    projected_q: []f32,
    projected_k: []f32,
    projected_v: []f32,
    attention_context: []f32,
    position_embedding: []f32,
    token_hidden: []f32,
    normed_hidden: []f32,
    intermediate: []f32,
    scores: []f32,
    logits: []f32,
    empty_scratch: [0]u8 = .{},

    fn init(allocator: std.mem.Allocator, cfg: decoder_types.DecoderConfig) !Workspace {
        const hidden_bytes = cfg.max_position_embeddings * cfg.hidden_size;
        return .{
            .hidden_a = try allocator.alloc(f32, hidden_bytes),
            .hidden_b = try allocator.alloc(f32, hidden_bytes),
            .projected_q = try allocator.alloc(f32, hidden_bytes),
            .projected_k = try allocator.alloc(f32, hidden_bytes),
            .projected_v = try allocator.alloc(f32, hidden_bytes),
            .attention_context = try allocator.alloc(f32, hidden_bytes),
            .position_embedding = try allocator.alloc(f32, cfg.hidden_size),
            .token_hidden = try allocator.alloc(f32, cfg.hidden_size),
            .normed_hidden = try allocator.alloc(f32, cfg.hidden_size),
            .intermediate = try allocator.alloc(f32, cfg.intermediate_size),
            .scores = try allocator.alloc(f32, cfg.max_position_embeddings * cfg.max_position_embeddings),
            .logits = try allocator.alloc(f32, cfg.vocab_size),
        };
    }

    fn deinit(self: *Workspace, allocator: std.mem.Allocator) void {
        allocator.free(self.hidden_a);
        allocator.free(self.hidden_b);
        allocator.free(self.projected_q);
        allocator.free(self.projected_k);
        allocator.free(self.projected_v);
        allocator.free(self.attention_context);
        allocator.free(self.position_embedding);
        allocator.free(self.token_hidden);
        allocator.free(self.normed_hidden);
        allocator.free(self.intermediate);
        allocator.free(self.scores);
        allocator.free(self.logits);
    }
};

const LayerWeights = struct {
    query_tensor: tensor_backend.Backend.TensorHandle,
    key_tensor: tensor_backend.Backend.TensorHandle,
    value_tensor: tensor_backend.Backend.TensorHandle,
    attention_output_tensor: tensor_backend.Backend.TensorHandle,
    intermediate_tensor: tensor_backend.Backend.TensorHandle,
    output_tensor: tensor_backend.Backend.TensorHandle,
    query_bias: []f32,
    key_bias: []f32,
    value_bias: []f32,
    attention_output_bias: []f32,
    attention_norm_weight: []f32,
    attention_norm_bias: []f32,
    intermediate_bias: []f32,
    output_bias: []f32,
    output_norm_weight: []f32,
    output_norm_bias: []f32,

    fn init(
        allocator: std.mem.Allocator,
        backend: *const tensor_backend.Backend,
        layer_index: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) !LayerWeights {
        return .{
            .query_tensor = try resolveLayerTensor(allocator, backend, layer_index, "attention.self.query.weight"),
            .key_tensor = try resolveLayerTensor(allocator, backend, layer_index, "attention.self.key.weight"),
            .value_tensor = try resolveLayerTensor(allocator, backend, layer_index, "attention.self.value.weight"),
            .attention_output_tensor = try resolveLayerTensor(allocator, backend, layer_index, "attention.output.dense.weight"),
            .intermediate_tensor = try resolveLayerTensor(allocator, backend, layer_index, "intermediate.dense.weight"),
            .output_tensor = try resolveLayerTensor(allocator, backend, layer_index, "output.dense.weight"),
            .query_bias = try loadLayerVector(allocator, backend, layer_index, "attention.self.query.bias", hidden_size),
            .key_bias = try loadLayerVector(allocator, backend, layer_index, "attention.self.key.bias", hidden_size),
            .value_bias = try loadLayerVector(allocator, backend, layer_index, "attention.self.value.bias", hidden_size),
            .attention_output_bias = try loadLayerVector(allocator, backend, layer_index, "attention.output.dense.bias", hidden_size),
            .attention_norm_weight = try loadLayerVector(allocator, backend, layer_index, "attention.output.LayerNorm.gamma", hidden_size),
            .attention_norm_bias = try loadLayerVector(allocator, backend, layer_index, "attention.output.LayerNorm.beta", hidden_size),
            .intermediate_bias = try loadLayerVector(allocator, backend, layer_index, "intermediate.dense.bias", intermediate_size),
            .output_bias = try loadLayerVector(allocator, backend, layer_index, "output.dense.bias", hidden_size),
            .output_norm_weight = try loadLayerVector(allocator, backend, layer_index, "output.LayerNorm.gamma", hidden_size),
            .output_norm_bias = try loadLayerVector(allocator, backend, layer_index, "output.LayerNorm.beta", hidden_size),
        };
    }

    fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.query_bias);
        allocator.free(self.key_bias);
        allocator.free(self.value_bias);
        allocator.free(self.attention_output_bias);
        allocator.free(self.attention_norm_weight);
        allocator.free(self.attention_norm_bias);
        allocator.free(self.intermediate_bias);
        allocator.free(self.output_bias);
        allocator.free(self.output_norm_weight);
        allocator.free(self.output_norm_bias);
    }
};

pub const Runtime = struct {
    allocator: std.mem.Allocator,
    cfg: decoder_types.DecoderConfig,
    tokenizer: bert_family.TokenizerImpl,
    backend: tensor_backend.Backend,
    parallel_pool: parallel_rows.Pool,
    thread_count: usize,
    workspace: Workspace,
    word_embeddings_tensor: tensor_backend.Backend.TensorHandle,
    position_embeddings_tensor: tensor_backend.Backend.TensorHandle,
    mlm_transform_tensor: tensor_backend.Backend.TensorHandle,
    layers: []LayerWeights,
    embeddings_norm_weight: []f32,
    embeddings_norm_bias: []f32,
    token_type_zero_embedding: []f32,
    mlm_transform_bias: []f32,
    mlm_norm_weight: []f32,
    mlm_norm_bias: []f32,
    mlm_bias: []f32,
    cls_token_id: u32,
    sep_token_id: u32,
    mask_token_id: u32,

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8) !Runtime {
        return initWithThreads(allocator, model_dir, null);
    }

    pub fn initWithThreads(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        requested_thread_count: ?usize,
    ) !Runtime {
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);

        var parsed_config = try bert_family.loadParsedConfig(allocator, config_path);
        errdefer parsed_config.deinit();

        var tokenizer = try bert_family.loadTokenizerFromModelDir(allocator, model_dir);
        errdefer tokenizer.deinit();

        var backend = try tensor_backend.Backend.openFromModelDir(allocator, model_dir, .auto);
        errdefer backend.deinit();

        const cfg = parsed_config.value;
        const resolved_thread_count = if (requested_thread_count) |count|
            @max(@as(usize, 1), count)
        else
            @max(@as(usize, 1), std.Thread.getCpuCount() catch 1);
        var parallel_pool = try parallel_rows.Pool.init(allocator, resolved_thread_count);
        errdefer parallel_pool.deinit();
        var workspace = try Workspace.init(allocator, cfg);
        errdefer workspace.deinit(allocator);

        const word_embeddings_tensor = try backend.resolveTensor(word_embeddings_name);
        const position_embeddings_tensor = try backend.resolveTensor(position_embeddings_name);
        const mlm_transform_tensor = try backend.resolveTensor(mlm_transform_weight_name);

        const embeddings_norm_weight = try loadVectorByName(allocator, &backend, "bert.embeddings.LayerNorm.gamma", cfg.hidden_size);
        errdefer allocator.free(embeddings_norm_weight);
        const embeddings_norm_bias = try loadVectorByName(allocator, &backend, "bert.embeddings.LayerNorm.beta", cfg.hidden_size);
        errdefer allocator.free(embeddings_norm_bias);
        const token_type_zero_embedding = try loadRowByName(allocator, &backend, "bert.embeddings.token_type_embeddings.weight", 0, cfg.hidden_size);
        errdefer allocator.free(token_type_zero_embedding);
        const mlm_transform_bias = try loadVectorByName(allocator, &backend, "cls.predictions.transform.dense.bias", cfg.hidden_size);
        errdefer allocator.free(mlm_transform_bias);
        const mlm_norm_weight = try loadVectorByName(allocator, &backend, "cls.predictions.transform.LayerNorm.gamma", cfg.hidden_size);
        errdefer allocator.free(mlm_norm_weight);
        const mlm_norm_bias = try loadVectorByName(allocator, &backend, "cls.predictions.transform.LayerNorm.beta", cfg.hidden_size);
        errdefer allocator.free(mlm_norm_bias);
        const mlm_bias = try loadVectorByName(allocator, &backend, "cls.predictions.bias", cfg.vocab_size);
        errdefer allocator.free(mlm_bias);

        const cls_token_id = tokenizer.idForToken(tokenizer.cls_token) orelse return error.MissingClsToken;
        const sep_token_id = tokenizer.idForToken(tokenizer.sep_token) orelse return error.MissingSepToken;
        const mask_token_id = tokenizer.idForToken(tokenizer.mask_token) orelse return error.MissingMaskToken;

        const layers = try allocator.alloc(LayerWeights, cfg.num_hidden_layers);
        errdefer allocator.free(layers);
        var initialized: usize = 0;
        errdefer {
            for (layers[0..initialized]) |*layer| layer.deinit(allocator);
        }
        for (layers, 0..) |*layer, layer_index| {
            layer.* = try LayerWeights.init(allocator, &backend, layer_index, cfg.hidden_size, cfg.intermediate_size);
            initialized += 1;
        }

        parsed_config.deinit();
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .tokenizer = tokenizer,
            .backend = backend,
            .parallel_pool = parallel_pool,
            .thread_count = resolved_thread_count,
            .workspace = workspace,
            .word_embeddings_tensor = word_embeddings_tensor,
            .position_embeddings_tensor = position_embeddings_tensor,
            .mlm_transform_tensor = mlm_transform_tensor,
            .layers = layers,
            .embeddings_norm_weight = embeddings_norm_weight,
            .embeddings_norm_bias = embeddings_norm_bias,
            .token_type_zero_embedding = token_type_zero_embedding,
            .mlm_transform_bias = mlm_transform_bias,
            .mlm_norm_weight = mlm_norm_weight,
            .mlm_norm_bias = mlm_norm_bias,
            .mlm_bias = mlm_bias,
            .cls_token_id = cls_token_id,
            .sep_token_id = sep_token_id,
            .mask_token_id = mask_token_id,
        };
    }

    pub fn deinit(self: *Runtime) void {
        for (self.layers) |*layer| layer.deinit(self.allocator);
        self.allocator.free(self.layers);
        self.allocator.free(self.embeddings_norm_weight);
        self.allocator.free(self.embeddings_norm_bias);
        self.allocator.free(self.token_type_zero_embedding);
        self.allocator.free(self.mlm_transform_bias);
        self.allocator.free(self.mlm_norm_weight);
        self.allocator.free(self.mlm_norm_bias);
        self.allocator.free(self.mlm_bias);
        self.workspace.deinit(self.allocator);
        self.parallel_pool.deinit();
        self.backend.deinit();
        self.tokenizer.deinit();
    }

    pub fn fillMask(self: *Runtime, text: []const u8, top_k: usize) !FillMaskResult {
        if (top_k == 0) return error.InvalidTopK;

        const encoded = try self.tokenizer.encodeAlloc(self.allocator, text);
        defer self.allocator.free(encoded);

        var mask_count: usize = 0;
        var mask_index: usize = 0;
        for (encoded, 0..) |token_id, idx| {
            if (token_id == self.mask_token_id) {
                mask_count += 1;
                mask_index = idx;
            }
        }
        if (mask_count != 1) return error.InvalidMaskCount;

        const input_ids = try self.wrapEncodedWithSpecialTokensAlloc(encoded);
        defer self.allocator.free(input_ids);
        const mask_position = mask_index + 1;
        const encoded_output = try self.runEncoder(input_ids);

        const masked_hidden = encoded_output.hidden[mask_position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
        try self.backend.matmulVec(
            self.workspace.token_hidden,
            self.mlm_transform_tensor,
            masked_hidden,
            self.thread_count,
            &self.parallel_pool,
            self.workspace.empty_scratch[0..],
        );
        addBiasInPlace(self.workspace.token_hidden, self.mlm_transform_bias);
        cpu.geluInPlace(self.workspace.token_hidden);
        try cpu.layerNorm(
            self.workspace.token_hidden,
            self.workspace.token_hidden,
            self.mlm_norm_weight,
            self.mlm_norm_bias,
            @floatCast(self.cfg.rms_norm_eps),
        );

        try self.backend.matmulVec(
            self.workspace.logits,
            self.word_embeddings_tensor,
            self.workspace.token_hidden,
            self.thread_count,
            &self.parallel_pool,
            self.workspace.empty_scratch[0..],
        );
        addBiasInPlace(self.workspace.logits, self.mlm_bias);

        const top = try logits_util.topKLogitsAlloc(self.allocator, self.workspace.logits, top_k);
        defer self.allocator.free(top);

        const predictions = try self.allocator.alloc(Prediction, top.len);
        errdefer self.allocator.free(predictions);

        for (top, 0..) |entry, idx| {
            const token_id = std.math.cast(u32, entry.token_id) orelse return error.TokenIdOutOfRange;
            const token = self.tokenizer.tokenForId(token_id) orelse return error.UnknownTokenId;
            predictions[idx] = .{
                .token_id = token_id,
                .token = try self.allocator.dupe(u8, token),
                .logit = entry.logit,
            };
        }

        return .{
            .mask_position = mask_position,
            .predictions = predictions,
        };
    }

    pub fn embedText(self: *Runtime, text: []const u8, mode: EmbeddingMode) ![]f32 {
        const encoded = try self.tokenizer.encodeAlloc(self.allocator, text);
        defer self.allocator.free(encoded);

        const input_ids = try self.wrapEncodedWithSpecialTokensAlloc(encoded);
        defer self.allocator.free(input_ids);
        const encoded_output = try self.runEncoder(input_ids);

        const embedding = try self.allocator.alloc(f32, self.cfg.hidden_size);
        switch (mode) {
            .cls => {
                @memcpy(embedding, encoded_output.hidden[0..self.cfg.hidden_size]);
            },
            .mean => {
                @memset(embedding, 0.0);
                const start: usize = if (encoded_output.seq_len > 2) 1 else 0;
                const end: usize = if (encoded_output.seq_len > 2) encoded_output.seq_len - 1 else encoded_output.seq_len;
                const token_count = @max(@as(usize, 1), end - start);
                for (start..end) |position| {
                    const hidden_slice = encoded_output.hidden[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
                    for (embedding, hidden_slice) |*out, value| out.* += value;
                }
                const scale = 1.0 / @as(f32, @floatFromInt(token_count));
                for (embedding) |*value| value.* *= scale;
            },
        }
        return embedding;
    }

    fn wrapEncodedWithSpecialTokensAlloc(self: *Runtime, encoded: []const u32) ![]usize {
        const seq_len = encoded.len + 2;
        if (seq_len > self.cfg.max_position_embeddings) return error.SequenceTooLong;

        const input_ids = try self.allocator.alloc(usize, seq_len);
        input_ids[0] = self.cls_token_id;
        for (encoded, 0..) |token_id, idx| {
            input_ids[idx + 1] = token_id;
        }
        input_ids[seq_len - 1] = self.sep_token_id;
        return input_ids;
    }

    fn runEncoder(self: *Runtime, input_ids: []const usize) !EncoderOutput {
        const seq_len = input_ids.len;
        const hidden_len = seq_len * self.cfg.hidden_size;

        const hidden_a = self.workspace.hidden_a[0..hidden_len];
        const hidden_b = self.workspace.hidden_b[0..hidden_len];
        const projected_q = self.workspace.projected_q[0..hidden_len];
        const projected_k = self.workspace.projected_k[0..hidden_len];
        const projected_v = self.workspace.projected_v[0..hidden_len];
        const attention_context = self.workspace.attention_context[0..hidden_len];
        const scores = self.workspace.scores[0 .. seq_len * seq_len];

        try self.embedInputs(input_ids, hidden_a);

        var hidden_in = hidden_a;
        var hidden_out = hidden_b;
        for (self.layers) |*layer| {
            try self.forwardLayer(
                layer,
                seq_len,
                hidden_in,
                hidden_out,
                projected_q,
                projected_k,
                projected_v,
                attention_context,
                scores,
            );
            std.mem.swap([]f32, &hidden_in, &hidden_out);
        }

        return .{
            .hidden = hidden_in,
            .seq_len = seq_len,
        };
    }

    fn embedInputs(self: *Runtime, input_ids: []const usize, hidden: []f32) !void {
        for (input_ids, 0..) |token_id, position| {
            const hidden_slice = hidden[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            try self.backend.readRowIntoTensor(
                self.word_embeddings_tensor,
                token_id,
                hidden_slice,
                self.workspace.empty_scratch[0..],
            );
            try self.backend.readRowIntoTensor(
                self.position_embeddings_tensor,
                position,
                self.workspace.position_embedding,
                self.workspace.empty_scratch[0..],
            );
            try cpu.axpyInPlace(hidden_slice, 1.0, self.token_type_zero_embedding);
            try cpu.axpyInPlace(hidden_slice, 1.0, self.workspace.position_embedding);
            try cpu.layerNorm(
                hidden_slice,
                hidden_slice,
                self.embeddings_norm_weight,
                self.embeddings_norm_bias,
                @floatCast(self.cfg.rms_norm_eps),
            );
        }
    }

    fn forwardLayer(
        self: *Runtime,
        layer: *const LayerWeights,
        seq_len: usize,
        hidden_in: []const f32,
        hidden_out: []f32,
        projected_q: []f32,
        projected_k: []f32,
        projected_v: []f32,
        attention_context: []f32,
        scores: []f32,
    ) !void {
        for (0..seq_len) |position| {
            const input_slice = hidden_in[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const q_slice = projected_q[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const k_slice = projected_k[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const v_slice = projected_v[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];

            try self.backend.matmulVec(q_slice, layer.query_tensor, input_slice, self.thread_count, &self.parallel_pool, self.workspace.empty_scratch[0..]);
            addBiasInPlace(q_slice, layer.query_bias);
            try self.backend.matmulVec(k_slice, layer.key_tensor, input_slice, self.thread_count, &self.parallel_pool, self.workspace.empty_scratch[0..]);
            addBiasInPlace(k_slice, layer.key_bias);
            try self.backend.matmulVec(v_slice, layer.value_tensor, input_slice, self.thread_count, &self.parallel_pool, self.workspace.empty_scratch[0..]);
            addBiasInPlace(v_slice, layer.value_bias);
        }

        try self.computeAttention(seq_len, projected_q, projected_k, projected_v, attention_context, scores);

        for (0..seq_len) |position| {
            const input_slice = hidden_in[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const output_slice = hidden_out[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const context_slice = attention_context[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];

            try self.backend.matmulVec(
                self.workspace.token_hidden,
                layer.attention_output_tensor,
                context_slice,
                self.thread_count,
                &self.parallel_pool,
                self.workspace.empty_scratch[0..],
            );
            addBiasInPlace(self.workspace.token_hidden, layer.attention_output_bias);
            try cpu.axpyInPlace(self.workspace.token_hidden, 1.0, input_slice);
            try cpu.layerNorm(
                self.workspace.normed_hidden,
                self.workspace.token_hidden,
                layer.attention_norm_weight,
                layer.attention_norm_bias,
                @floatCast(self.cfg.rms_norm_eps),
            );

            try self.backend.matmulVec(
                self.workspace.intermediate,
                layer.intermediate_tensor,
                self.workspace.normed_hidden,
                self.thread_count,
                &self.parallel_pool,
                self.workspace.empty_scratch[0..],
            );
            addBiasInPlace(self.workspace.intermediate, layer.intermediate_bias);
            cpu.geluInPlace(self.workspace.intermediate);

            try self.backend.matmulVec(
                self.workspace.token_hidden,
                layer.output_tensor,
                self.workspace.intermediate,
                self.thread_count,
                &self.parallel_pool,
                self.workspace.empty_scratch[0..],
            );
            addBiasInPlace(self.workspace.token_hidden, layer.output_bias);
            try cpu.axpyInPlace(self.workspace.token_hidden, 1.0, self.workspace.normed_hidden);
            try cpu.layerNorm(
                output_slice,
                self.workspace.token_hidden,
                layer.output_norm_weight,
                layer.output_norm_bias,
                @floatCast(self.cfg.rms_norm_eps),
            );
        }
    }

    fn computeAttention(
        self: *Runtime,
        seq_len: usize,
        projected_q: []const f32,
        projected_k: []const f32,
        projected_v: []const f32,
        output: []f32,
        scores: []f32,
    ) !void {
        const Context = struct {
            runtime: *Runtime,
            seq_len: usize,
            projected_q: []const f32,
            projected_k: []const f32,
            projected_v: []const f32,
            output: []f32,
            scores: []f32,

            fn runRange(ctx_ptr: *anyopaque, start_row: usize, end_row: usize) void {
                const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                ctx.runtime.computeAttentionRange(
                    ctx.seq_len,
                    ctx.projected_q,
                    ctx.projected_k,
                    ctx.projected_v,
                    ctx.output,
                    ctx.scores,
                    start_row,
                    end_row,
                ) catch unreachable;
            }
        };

        if (shouldParallelizeAttention(seq_len, self.cfg.hidden_size, self.thread_count, self.parallel_pool.workerCount() > 1)) {
            var context = Context{
                .runtime = self,
                .seq_len = seq_len,
                .projected_q = projected_q,
                .projected_k = projected_k,
                .projected_v = projected_v,
                .output = output,
                .scores = scores,
            };
            self.parallel_pool.run(seq_len, &context, Context.runRange);
            return;
        }

        try self.computeAttentionRange(seq_len, projected_q, projected_k, projected_v, output, scores, 0, seq_len);
    }

    fn computeAttentionRange(
        self: *Runtime,
        seq_len: usize,
        projected_q: []const f32,
        projected_k: []const f32,
        projected_v: []const f32,
        output: []f32,
        scores: []f32,
        start_query: usize,
        end_query: usize,
    ) !void {
        const head_count = self.cfg.num_attention_heads;
        const head_dim = self.cfg.head_dim;
        const hidden_size = self.cfg.hidden_size;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (start_query..end_query) |query_position| {
            const q_base = query_position * hidden_size;
            const out_slice = output[q_base..][0..hidden_size];
            const score_slice = scores[query_position * seq_len ..][0..seq_len];
            @memset(out_slice, 0.0);

            for (0..head_count) |head_index| {
                const head_offset = head_index * head_dim;
                const query_head = projected_q[q_base + head_offset ..][0..head_dim];
                for (0..seq_len) |key_position| {
                    const key_head = projected_k[key_position * hidden_size + head_offset ..][0..head_dim];
                    score_slice[key_position] = dotHot(query_head, key_head) * scale;
                }
                try attention.softmaxInPlace(score_slice);

                const out_head = out_slice[head_offset..][0..head_dim];
                for (0..seq_len) |key_position| {
                    const value_head = projected_v[key_position * hidden_size + head_offset ..][0..head_dim];
                    axpyHot(out_head, score_slice[key_position], value_head);
                }
            }
        }
    }
};

fn dotHot(lhs: []const f32, rhs: []const f32) f32 {
    std.debug.assert(lhs.len == rhs.len);

    var acc0: @Vector(8, f32) = @splat(0.0);
    var acc1: @Vector(8, f32) = @splat(0.0);
    var index: usize = 0;
    while (index + 16 <= lhs.len) : (index += 16) {
        const lhs0: @Vector(8, f32) = lhs[index..][0..8].*;
        const rhs0: @Vector(8, f32) = rhs[index..][0..8].*;
        const lhs1: @Vector(8, f32) = lhs[index + 8 ..][0..8].*;
        const rhs1: @Vector(8, f32) = rhs[index + 8 ..][0..8].*;
        acc0 += lhs0 * rhs0;
        acc1 += lhs1 * rhs1;
    }

    var sum = @reduce(.Add, acc0 + acc1);
    while (index < lhs.len) : (index += 1) {
        sum += lhs[index] * rhs[index];
    }
    return sum;
}

fn axpyHot(output: []f32, alpha: f32, input: []const f32) void {
    std.debug.assert(output.len == input.len);

    const alpha_vec: @Vector(8, f32) = @splat(alpha);
    var index: usize = 0;
    while (index + 8 <= output.len) : (index += 8) {
        const out_vec: @Vector(8, f32) = output[index..][0..8].*;
        const in_vec: @Vector(8, f32) = input[index..][0..8].*;
        output[index..][0..8].* = out_vec + alpha_vec * in_vec;
    }
    while (index < output.len) : (index += 1) {
        output[index] += alpha * input[index];
    }
}

fn addBiasInPlace(values: []f32, bias: []const f32) void {
    std.debug.assert(values.len == bias.len);
    for (values, bias) |*value, bias_value| {
        value.* += bias_value;
    }
}

fn resolveLayerTensor(
    allocator: std.mem.Allocator,
    backend: *const tensor_backend.Backend,
    layer_index: usize,
    suffix: []const u8,
) !tensor_backend.Backend.TensorHandle {
    const name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.{s}", .{ layer_index, suffix });
    defer allocator.free(name);
    return try backend.resolveTensor(name);
}

fn loadLayerVector(
    allocator: std.mem.Allocator,
    backend: *const tensor_backend.Backend,
    layer_index: usize,
    suffix: []const u8,
    count: usize,
) ![]f32 {
    const name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.{s}", .{ layer_index, suffix });
    defer allocator.free(name);
    return try loadVectorByName(allocator, backend, name, count);
}

fn loadVectorByName(
    allocator: std.mem.Allocator,
    backend: *const tensor_backend.Backend,
    name: []const u8,
    count: usize,
) ![]f32 {
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    try backend.readVectorInto(name, values, &.{});
    return values;
}

fn loadRowByName(
    allocator: std.mem.Allocator,
    backend: *const tensor_backend.Backend,
    name: []const u8,
    row_index: usize,
    row_width: usize,
) ![]f32 {
    const values = try allocator.alloc(f32, row_width);
    errdefer allocator.free(values);
    try backend.readRowInto(name, row_index, values, &.{});
    return values;
}

fn shouldParallelizeAttention(seq_len: usize, hidden_size: usize, thread_count: usize, has_pool: bool) bool {
    if (!has_pool or thread_count <= 1) return false;
    const work = std.math.mul(u64, seq_len * seq_len, hidden_size) catch return true;
    return work >= 131_072;
}
