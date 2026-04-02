const std = @import("std");
const cpu = @import("../../kernel/core/cpu.zig");
const attention = @import("../../kernel/attention/basic.zig");
const logits_util = @import("../layers/logits.zig");
const tensor_store = @import("../../tensor/storage/store.zig");
const bert_family = @import("../families/bert/family.zig");
const decoder_types = @import("decoder_types.zig");

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

const LayerWeights = struct {
    query_weight_name: []u8,
    key_weight_name: []u8,
    value_weight_name: []u8,
    attention_output_weight_name: []u8,
    intermediate_weight_name: []u8,
    output_weight_name: []u8,
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
        store: *const tensor_store.TensorStore,
        layer_index: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) !LayerWeights {
        return .{
            .query_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.attention.self.query.weight", .{layer_index}),
            .key_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.attention.self.key.weight", .{layer_index}),
            .value_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.attention.self.value.weight", .{layer_index}),
            .attention_output_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.attention.output.dense.weight", .{layer_index}),
            .intermediate_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.intermediate.dense.weight", .{layer_index}),
            .output_weight_name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.output.dense.weight", .{layer_index}),
            .query_bias = try loadVector(allocator, store, layer_index, "attention.self.query.bias", hidden_size),
            .key_bias = try loadVector(allocator, store, layer_index, "attention.self.key.bias", hidden_size),
            .value_bias = try loadVector(allocator, store, layer_index, "attention.self.value.bias", hidden_size),
            .attention_output_bias = try loadVector(allocator, store, layer_index, "attention.output.dense.bias", hidden_size),
            .attention_norm_weight = try loadVector(allocator, store, layer_index, "attention.output.LayerNorm.gamma", hidden_size),
            .attention_norm_bias = try loadVector(allocator, store, layer_index, "attention.output.LayerNorm.beta", hidden_size),
            .intermediate_bias = try loadVector(allocator, store, layer_index, "intermediate.dense.bias", intermediate_size),
            .output_bias = try loadVector(allocator, store, layer_index, "output.dense.bias", hidden_size),
            .output_norm_weight = try loadVector(allocator, store, layer_index, "output.LayerNorm.gamma", hidden_size),
            .output_norm_bias = try loadVector(allocator, store, layer_index, "output.LayerNorm.beta", hidden_size),
        };
    }

    fn deinit(self: *LayerWeights, allocator: std.mem.Allocator) void {
        allocator.free(self.query_weight_name);
        allocator.free(self.key_weight_name);
        allocator.free(self.value_weight_name);
        allocator.free(self.attention_output_weight_name);
        allocator.free(self.intermediate_weight_name);
        allocator.free(self.output_weight_name);
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
    store: tensor_store.TensorStore,
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
        const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
        defer allocator.free(config_path);

        var parsed_config = try bert_family.loadParsedConfig(allocator, config_path);
        errdefer parsed_config.deinit();

        var tokenizer = try bert_family.loadTokenizerFromModelDir(allocator, model_dir);
        errdefer tokenizer.deinit();

        const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
        defer allocator.free(weights_path);
        var store = try tensor_store.TensorStore.open(allocator, weights_path);
        errdefer store.deinit();

        const cfg = parsed_config.value;
        const embeddings_norm_weight = try store.readElementsAsF32Alloc("bert.embeddings.LayerNorm.gamma", 0, cfg.hidden_size);
        errdefer allocator.free(embeddings_norm_weight);
        const embeddings_norm_bias = try store.readElementsAsF32Alloc("bert.embeddings.LayerNorm.beta", 0, cfg.hidden_size);
        errdefer allocator.free(embeddings_norm_bias);
        const token_type_zero_embedding = try store.readRowAsF32Alloc("bert.embeddings.token_type_embeddings.weight", 0);
        errdefer allocator.free(token_type_zero_embedding);
        const mlm_transform_bias = try store.readElementsAsF32Alloc("cls.predictions.transform.dense.bias", 0, cfg.hidden_size);
        errdefer allocator.free(mlm_transform_bias);
        const mlm_norm_weight = try store.readElementsAsF32Alloc("cls.predictions.transform.LayerNorm.gamma", 0, cfg.hidden_size);
        errdefer allocator.free(mlm_norm_weight);
        const mlm_norm_bias = try store.readElementsAsF32Alloc("cls.predictions.transform.LayerNorm.beta", 0, cfg.hidden_size);
        errdefer allocator.free(mlm_norm_bias);
        const mlm_bias = try store.readElementsAsF32Alloc("cls.predictions.bias", 0, cfg.vocab_size);
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
            layer.* = try LayerWeights.init(allocator, &store, layer_index, cfg.hidden_size, cfg.intermediate_size);
            initialized += 1;
        }

        parsed_config.deinit();
        return .{
            .allocator = allocator,
            .cfg = cfg,
            .tokenizer = tokenizer,
            .store = store,
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
        self.store.deinit();
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

        const seq_len = encoded.len + 2;
        if (seq_len > self.cfg.max_position_embeddings) return error.SequenceTooLong;

        const input_ids = try self.allocator.alloc(usize, seq_len);
        defer self.allocator.free(input_ids);
        input_ids[0] = self.cls_token_id;
        for (encoded, 0..) |token_id, idx| {
            input_ids[idx + 1] = token_id;
        }
        input_ids[seq_len - 1] = self.sep_token_id;
        const mask_position = mask_index + 1;

        const hidden_bytes = seq_len * self.cfg.hidden_size;
        const hidden_a = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(hidden_a);
        const hidden_b = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(hidden_b);
        const projected_q = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(projected_q);
        const projected_k = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(projected_k);
        const projected_v = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(projected_v);
        const attention_context = try self.allocator.alloc(f32, hidden_bytes);
        defer self.allocator.free(attention_context);
        const position_embedding = try self.allocator.alloc(f32, self.cfg.hidden_size);
        defer self.allocator.free(position_embedding);
        const token_hidden = try self.allocator.alloc(f32, self.cfg.hidden_size);
        defer self.allocator.free(token_hidden);
        const normed_hidden = try self.allocator.alloc(f32, self.cfg.hidden_size);
        defer self.allocator.free(normed_hidden);
        const intermediate = try self.allocator.alloc(f32, self.cfg.intermediate_size);
        defer self.allocator.free(intermediate);
        const scores = try self.allocator.alloc(f32, seq_len);
        defer self.allocator.free(scores);
        const logits = try self.allocator.alloc(f32, self.cfg.vocab_size);
        defer self.allocator.free(logits);

        try self.embedInputs(input_ids, hidden_a, position_embedding);

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
                token_hidden,
                normed_hidden,
                intermediate,
                scores,
            );
            std.mem.swap([]f32, &hidden_in, &hidden_out);
        }

        const masked_hidden = hidden_in[mask_position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
        try self.store.matmulVecByName(token_hidden, mlm_transform_weight_name, masked_hidden);
        addBiasInPlace(token_hidden, self.mlm_transform_bias);
        cpu.geluInPlace(token_hidden);
        try cpu.layerNorm(token_hidden, token_hidden, self.mlm_norm_weight, self.mlm_norm_bias, @floatCast(self.cfg.rms_norm_eps));

        try self.store.matmulVecByName(logits, word_embeddings_name, token_hidden);
        addBiasInPlace(logits, self.mlm_bias);

        const top = try logits_util.topKLogitsAlloc(self.allocator, logits, top_k);
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

    fn embedInputs(
        self: *Runtime,
        input_ids: []const usize,
        hidden: []f32,
        position_embedding: []f32,
    ) !void {
        for (input_ids, 0..) |token_id, position| {
            const hidden_slice = hidden[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            try self.store.readRowAsF32Into(word_embeddings_name, token_id, hidden_slice, &.{});
            try self.store.readRowAsF32Into(position_embeddings_name, position, position_embedding, &.{});
            try cpu.axpyInPlace(hidden_slice, 1.0, self.token_type_zero_embedding);
            try cpu.axpyInPlace(hidden_slice, 1.0, position_embedding);
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
        token_hidden: []f32,
        normed_hidden: []f32,
        intermediate: []f32,
        scores: []f32,
    ) !void {
        for (0..seq_len) |position| {
            const input_slice = hidden_in[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const q_slice = projected_q[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const k_slice = projected_k[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const v_slice = projected_v[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];

            try self.store.matmulVecByName(q_slice, layer.query_weight_name, input_slice);
            addBiasInPlace(q_slice, layer.query_bias);
            try self.store.matmulVecByName(k_slice, layer.key_weight_name, input_slice);
            addBiasInPlace(k_slice, layer.key_bias);
            try self.store.matmulVecByName(v_slice, layer.value_weight_name, input_slice);
            addBiasInPlace(v_slice, layer.value_bias);
        }

        try self.computeAttention(seq_len, projected_q, projected_k, projected_v, attention_context, scores);

        for (0..seq_len) |position| {
            const input_slice = hidden_in[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const output_slice = hidden_out[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            const context_slice = attention_context[position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];

            try self.store.matmulVecByName(token_hidden, layer.attention_output_weight_name, context_slice);
            addBiasInPlace(token_hidden, layer.attention_output_bias);
            try cpu.axpyInPlace(token_hidden, 1.0, input_slice);
            try cpu.layerNorm(normed_hidden, token_hidden, layer.attention_norm_weight, layer.attention_norm_bias, @floatCast(self.cfg.rms_norm_eps));

            try self.store.matmulVecByName(intermediate, layer.intermediate_weight_name, normed_hidden);
            addBiasInPlace(intermediate, layer.intermediate_bias);
            cpu.geluInPlace(intermediate);

            try self.store.matmulVecByName(token_hidden, layer.output_weight_name, intermediate);
            addBiasInPlace(token_hidden, layer.output_bias);
            try cpu.axpyInPlace(token_hidden, 1.0, normed_hidden);
            try cpu.layerNorm(output_slice, token_hidden, layer.output_norm_weight, layer.output_norm_bias, @floatCast(self.cfg.rms_norm_eps));
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
        const head_count = self.cfg.num_attention_heads;
        const head_dim = self.cfg.head_dim;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..seq_len) |query_position| {
            const out_slice = output[query_position * self.cfg.hidden_size ..][0..self.cfg.hidden_size];
            @memset(out_slice, 0.0);

            for (0..head_count) |head_index| {
                const query_head = projected_q[query_position * self.cfg.hidden_size + head_index * head_dim ..][0..head_dim];
                for (0..seq_len) |key_position| {
                    const key_head = projected_k[key_position * self.cfg.hidden_size + head_index * head_dim ..][0..head_dim];
                    scores[key_position] = (try cpu.dot(query_head, key_head)) * scale;
                }
                try attention.softmaxInPlace(scores[0..seq_len]);

                const out_head = out_slice[head_index * head_dim ..][0..head_dim];
                for (0..seq_len) |key_position| {
                    const value_head = projected_v[key_position * self.cfg.hidden_size + head_index * head_dim ..][0..head_dim];
                    try cpu.axpyInPlace(out_head, scores[key_position], value_head);
                }
            }
        }
    }
};

fn addBiasInPlace(values: []f32, bias: []const f32) void {
    std.debug.assert(values.len == bias.len);
    for (values, bias) |*value, bias_value| {
        value.* += bias_value;
    }
}

fn loadVector(
    allocator: std.mem.Allocator,
    store: *const tensor_store.TensorStore,
    layer_index: usize,
    suffix: []const u8,
    count: usize,
) ![]f32 {
    const name = try std.fmt.allocPrint(allocator, "bert.encoder.layer.{d}.{s}", .{ layer_index, suffix });
    defer allocator.free(name);
    return try store.readElementsAsF32Alloc(name, 0, count);
}
