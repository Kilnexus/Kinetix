const std = @import("std");

pub const RoPETable = struct {
    allocator: std.mem.Allocator,
    max_seq_len: usize,
    head_dim: usize,
    cos: []f32,
    sin: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        max_seq_len: usize,
        head_dim: usize,
        rope_theta: f32,
    ) !RoPETable {
        if (head_dim == 0 or head_dim % 2 != 0) return error.InvalidHeadDim;
        const half = head_dim / 2;
        const total = try std.math.mul(usize, max_seq_len, half);

        const cos = try allocator.alloc(f32, total);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, total);
        errdefer allocator.free(sin);

        const dim_f = @as(f32, @floatFromInt(head_dim));
        for (0..max_seq_len) |position| {
            const pos_f = @as(f32, @floatFromInt(position));
            const base = position * half;
            for (0..half) |i| {
                const exponent = @as(f32, @floatFromInt(i * 2)) / dim_f;
                const inv_freq = 1.0 / std.math.pow(f32, rope_theta, exponent);
                const angle = pos_f * inv_freq;
                cos[base + i] = std.math.cos(angle);
                sin[base + i] = std.math.sin(angle);
            }
        }

        return .{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .cos = cos,
            .sin = sin,
        };
    }

    pub fn deinit(self: *RoPETable) void {
        self.allocator.free(self.cos);
        self.allocator.free(self.sin);
    }

    pub fn cosForPosition(self: *const RoPETable, position: usize) []const f32 {
        std.debug.assert(position < self.max_seq_len);
        const half = self.head_dim / 2;
        const start = position * half;
        return self.cos[start .. start + half];
    }

    pub fn sinForPosition(self: *const RoPETable, position: usize) []const f32 {
        std.debug.assert(position < self.max_seq_len);
        const half = self.head_dim / 2;
        const start = position * half;
        return self.sin[start .. start + half];
    }
};
