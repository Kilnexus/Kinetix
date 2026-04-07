const std = @import("std");
const gqa_attention = @import("../../../legacy/zinfer/src/model/layers/gqa_attention.zig");
const weights_layout = @import("weights_layout.zig");

pub const LayerTensorNameFn = *const fn (std.mem.Allocator, usize, weights_layout.LayerTensorKind) anyerror![]u8;

pub const LayerLayout = struct {
    input_layernorm_kind: weights_layout.LayerTensorKind = .input_layernorm_weight,
    q_norm_kind: ?weights_layout.LayerTensorKind = null,
    k_norm_kind: ?weights_layout.LayerTensorKind = null,
    q_proj_kind: weights_layout.LayerTensorKind = .self_attn_q_proj_weight,
    k_proj_kind: weights_layout.LayerTensorKind = .self_attn_k_proj_weight,
    v_proj_kind: weights_layout.LayerTensorKind = .self_attn_v_proj_weight,
    o_proj_kind: weights_layout.LayerTensorKind = .self_attn_o_proj_weight,
    post_attention_layernorm_kind: weights_layout.LayerTensorKind = .post_attention_layernorm_weight,
    gate_proj_kind: weights_layout.LayerTensorKind = .mlp_gate_proj_weight,
    up_proj_kind: weights_layout.LayerTensorKind = .mlp_up_proj_weight,
    down_proj_kind: weights_layout.LayerTensorKind = .mlp_down_proj_weight,
};

pub const Spec = struct {
    layer_index: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,

    pub fn validate(self: Spec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.intermediate_size == 0) return error.InvalidIntermediateSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
    }

    pub fn attentionSpec(self: Spec) gqa_attention.AttentionSpec {
        return .{
            .hidden_size = self.num_attention_heads * self.head_dim,
            .num_attention_heads = self.num_attention_heads,
            .num_key_value_heads = self.num_key_value_heads,
            .head_dim = self.head_dim,
            .rope_theta = self.rope_theta,
        };
    }
};

test "decoder block spec validates dimensions" {
    const testing = std.testing;

    const spec = Spec{
        .layer_index = 0,
        .hidden_size = 1024,
        .intermediate_size = 3072,
        .num_attention_heads = 16,
        .num_key_value_heads = 8,
        .head_dim = 64,
        .rope_theta = 1000000.0,
        .rms_norm_eps = 1e-6,
    };
    try spec.validate();
    try testing.expectEqual(@as(usize, 1024), spec.attentionSpec().hidden_size);
}
