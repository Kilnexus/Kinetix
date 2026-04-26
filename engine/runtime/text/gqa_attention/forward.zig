const attention = @import("../attention/attention.zig");
const decoder_types = @import("../decoder_types.zig");
const spec_mod = @import("spec.zig");

pub fn applyRoPEToProjectedHeadsInPlace(
    spec: spec_mod.AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try attention.applyRoPEToHeadsInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
    try attention.applyRoPEToHeadsInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
}

pub fn applyRoPEToProjectedHeadsWithTableInPlace(
    spec: spec_mod.AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const attention.RoPETable,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try attention.applyRoPEToHeadsWithTableInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        table,
        position,
    );
    try attention.applyRoPEToHeadsWithTableInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        table,
        position,
    );
}

pub fn applyRoPEToProjectedHeadsWithPositionInPlace(
    spec: spec_mod.AttentionSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const attention.RoPETable,
    position: decoder_types.TokenPosition,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    switch (spec.rope_position_mode) {
        .scalar => {
            const scalar_position = if (position.mode == .mrope) position.scalar else position.scalar;
            try applyRoPEToProjectedHeadsWithTableInPlace(
                spec,
                projected_query,
                projected_key,
                table,
                scalar_position,
            );
        },
        .mrope => {
            try attention.applyRoPEToHeadsWithPositionInPlace(
                projected_query,
                spec.num_attention_heads,
                spec.head_dim,
                table,
                position,
                spec.mrope_sections,
            );
            try attention.applyRoPEToHeadsWithPositionInPlace(
                projected_key,
                spec.num_key_value_heads,
                spec.head_dim,
                table,
                position,
                spec.mrope_sections,
            );
        },
    }
}
