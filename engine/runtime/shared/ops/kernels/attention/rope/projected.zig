const apply = @import("apply.zig");
const table_mod = @import("table.zig");
const types = @import("types.zig");

pub const ProjectedHeadsSpec = types.ProjectedHeadsSpec;
pub const Position = types.Position;
pub const RoPETable = table_mod.RoPETable;

pub fn applyRoPEToProjectedHeadsInPlace(
    spec: ProjectedHeadsSpec,
    projected_query: []f32,
    projected_key: []f32,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try apply.applyRoPEToHeadsInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
    try apply.applyRoPEToHeadsInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        position,
        spec.rope_theta,
    );
}

pub fn applyRoPEToProjectedHeadsWithTableInPlace(
    spec: ProjectedHeadsSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const RoPETable,
    position: usize,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    try apply.applyRoPEToHeadsWithTableInPlace(
        projected_query,
        spec.num_attention_heads,
        spec.head_dim,
        table,
        position,
    );
    try apply.applyRoPEToHeadsWithTableInPlace(
        projected_key,
        spec.num_key_value_heads,
        spec.head_dim,
        table,
        position,
    );
}

pub fn applyRoPEToProjectedHeadsWithPositionInPlace(
    spec: ProjectedHeadsSpec,
    projected_query: []f32,
    projected_key: []f32,
    table: *const RoPETable,
    position: Position,
) !void {
    try spec.validate();
    if (projected_query.len != spec.num_attention_heads * spec.head_dim) return error.SizeMismatch;
    if (projected_key.len != spec.num_key_value_heads * spec.head_dim) return error.SizeMismatch;

    switch (spec.rope_position_mode) {
        .scalar => {
            try applyRoPEToProjectedHeadsWithTableInPlace(
                spec,
                projected_query,
                projected_key,
                table,
                position.scalar,
            );
        },
        .mrope => {
            try apply.applyRoPEToHeadsWithPositionInPlace(
                projected_query,
                spec.num_attention_heads,
                spec.head_dim,
                table,
                position,
                spec.mrope_sections,
            );
            try apply.applyRoPEToHeadsWithPositionInPlace(
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
