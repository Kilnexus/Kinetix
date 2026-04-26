pub const PositionMode = enum {
    scalar,
    mrope,

    pub fn name(self: PositionMode) []const u8 {
        return switch (self) {
            .scalar => "scalar",
            .mrope => "mrope",
        };
    }
};

pub const Position = struct {
    mode: PositionMode = .scalar,
    scalar: usize = 0,
    axes: [4]usize = .{ 0, 0, 0, 0 },

    pub fn scalarPosition(position: usize) Position {
        return .{
            .mode = .scalar,
            .scalar = position,
        };
    }

    pub fn mropePosition(axes: [4]usize) Position {
        return .{
            .mode = .mrope,
            .scalar = axes[3],
            .axes = axes,
        };
    }
};

pub const ProjectedHeadsSpec = struct {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    rope_position_mode: PositionMode = .scalar,
    mrope_sections: [4]u32 = .{ 0, 0, 0, 0 },

    pub fn validate(self: ProjectedHeadsSpec) !void {
        if (self.hidden_size == 0) return error.InvalidHiddenSize;
        if (self.num_attention_heads == 0) return error.InvalidAttentionHeads;
        if (self.num_key_value_heads == 0) return error.InvalidKeyValueHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.num_attention_heads % self.num_key_value_heads != 0) return error.InvalidGrouping;
        if (self.hidden_size != self.num_attention_heads * self.head_dim) return error.HiddenSizeMismatch;
    }
};
