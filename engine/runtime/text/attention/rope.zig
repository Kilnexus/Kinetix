const std = @import("std");
const decoder_types = @import("../decoder_types.zig");

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

pub fn applyRoPEToHeadInPlace(
    head: []f32,
    position: usize,
    rope_theta: f32,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;

    const dim_f = @as(f32, @floatFromInt(head.len));
    const pos_f = @as(f32, @floatFromInt(position));
    const half = head.len / 2;

    for (0..half) |i| {
        const exponent = @as(f32, @floatFromInt(i * 2)) / dim_f;
        const inv_freq = 1.0 / std.math.pow(f32, rope_theta, exponent);
        const angle = pos_f * inv_freq;
        const cos_angle = std.math.cos(angle);
        const sin_angle = std.math.sin(angle);

        const x0 = head[i];
        const x1 = head[i + half];
        head[i] = x0 * cos_angle - x1 * sin_angle;
        head[i + half] = x1 * cos_angle + x0 * sin_angle;
    }
}

pub fn applyRoPEToHeadWithTableInPlace(
    head: []f32,
    cos_table: []const f32,
    sin_table: []const f32,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;
    const half = head.len / 2;
    if (cos_table.len != half or sin_table.len != half) return error.SizeMismatch;

    for (0..half) |i| {
        const x0 = head[i];
        const x1 = head[i + half];
        const cos_angle = cos_table[i];
        const sin_angle = sin_table[i];
        head[i] = x0 * cos_angle - x1 * sin_angle;
        head[i + half] = x1 * cos_angle + x0 * sin_angle;
    }
}

pub fn applyRoPEToHeadsInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    position: usize,
    rope_theta: f32,
) !void {
    if (heads.len != num_heads * head_dim) return error.SizeMismatch;

    for (0..num_heads) |head_idx| {
        const start = head_idx * head_dim;
        try applyRoPEToHeadInPlace(heads[start .. start + head_dim], position, rope_theta);
    }
}

pub fn applyRoPEToHeadsWithTableInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    table: *const RoPETable,
    position: usize,
) !void {
    if (heads.len != num_heads * head_dim) return error.SizeMismatch;
    if (table.head_dim != head_dim) return error.SizeMismatch;
    if (position >= table.max_seq_len) return error.PositionOutOfBounds;

    const cos = table.cosForPosition(position);
    const sin = table.sinForPosition(position);
    for (0..num_heads) |head_idx| {
        const start = head_idx * head_dim;
        try applyRoPEToHeadWithTableInPlace(heads[start .. start + head_dim], cos, sin);
    }
}

pub fn applyRoPEToHeadWithPositionInPlace(
    head: []f32,
    table: *const RoPETable,
    position: decoder_types.TokenPosition,
    mrope_sections: [4]u32,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;
    if (table.head_dim != head.len) return error.SizeMismatch;

    if (position.mode == .scalar) {
        return applyRoPEToHeadWithTableInPlace(
            head,
            table.cosForPosition(position.scalar),
            table.sinForPosition(position.scalar),
        );
    }

    const half = head.len / 2;
    const sections = normalizeSections(half, mrope_sections);
    var offset: usize = 0;
    for (sections, 0..) |section_len, axis_index| {
        if (section_len == 0) continue;
        const axis_position = if (position.mode == .mrope) position.axes[axis_index] else position.scalar;
        if (axis_position >= table.max_seq_len) return error.PositionOutOfBounds;
        try applyRoPESectionWithTableInPlace(
            head,
            table.cosForPosition(axis_position),
            table.sinForPosition(axis_position),
            offset,
            section_len,
        );
        offset += section_len;
    }
}

pub fn applyRoPEToHeadsWithPositionInPlace(
    heads: []f32,
    num_heads: usize,
    head_dim: usize,
    table: *const RoPETable,
    position: decoder_types.TokenPosition,
    mrope_sections: [4]u32,
) !void {
    if (heads.len != num_heads * head_dim) return error.SizeMismatch;

    for (0..num_heads) |head_idx| {
        const start = head_idx * head_dim;
        try applyRoPEToHeadWithPositionInPlace(
            heads[start .. start + head_dim],
            table,
            position,
            mrope_sections,
        );
    }
}

fn applyRoPESectionWithTableInPlace(
    head: []f32,
    cos_table: []const f32,
    sin_table: []const f32,
    start: usize,
    len: usize,
) !void {
    if (head.len % 2 != 0) return error.InvalidHeadDim;
    const half = head.len / 2;
    if (start + len > half) return error.SizeMismatch;
    if (cos_table.len != half or sin_table.len != half) return error.SizeMismatch;

    for (0..len) |local_index| {
        const freq_index = start + local_index;
        const x0 = head[freq_index];
        const x1 = head[freq_index + half];
        const cos_angle = cos_table[freq_index];
        const sin_angle = sin_table[freq_index];
        head[freq_index] = x0 * cos_angle - x1 * sin_angle;
        head[freq_index + half] = x1 * cos_angle + x0 * sin_angle;
    }
}

fn normalizeSections(half_dim: usize, mrope_sections: [4]u32) [4]usize {
    var normalized: [4]usize = .{ 0, 0, 0, 0 };
    var remaining = half_dim;
    var last_active: usize = 0;
    var has_active = false;

    for (mrope_sections, 0..) |section, index| {
        if (remaining == 0) break;
        const requested = std.math.cast(usize, section) orelse remaining;
        const actual = @min(requested, remaining);
        normalized[index] = actual;
        if (actual != 0) {
            last_active = index;
            has_active = true;
        }
        remaining -= actual;
    }

    if (remaining != 0) {
        if (has_active) {
            normalized[last_active] += remaining;
        } else {
            normalized[0] = remaining;
        }
    }

    return normalized;
}

test "rope leaves head unchanged at position zero" {
    const testing = std.testing;

    var head = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try applyRoPEToHeadInPlace(&head, 0, 10000.0);

    try testing.expectEqual(@as(f32, 1.0), head[0]);
    try testing.expectEqual(@as(f32, 2.0), head[1]);
    try testing.expectEqual(@as(f32, 3.0), head[2]);
    try testing.expectEqual(@as(f32, 4.0), head[3]);
}

test "rope table path matches direct rope" {
    const testing = std.testing;

    var table = try RoPETable.init(testing.allocator, 4, 4, 10000.0);
    defer table.deinit();

    var direct = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var cached = direct;

    try applyRoPEToHeadInPlace(&direct, 2, 10000.0);
    try applyRoPEToHeadWithTableInPlace(&cached, table.cosForPosition(2), table.sinForPosition(2));

    for (direct, cached) |lhs, rhs| {
        try testing.expectApproxEqAbs(lhs, rhs, 1e-6);
    }
}

test "rope rotates first and second half pairs at nonzero position" {
    const testing = std.testing;

    var head = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try applyRoPEToHeadInPlace(&head, 1, 10000.0);

    const cos0 = std.math.cos(@as(f32, 1.0));
    const sin0 = std.math.sin(@as(f32, 1.0));
    const inv_freq1 = 1.0 / std.math.pow(f32, @as(f32, 10000.0), 0.5);
    const cos1 = std.math.cos(inv_freq1);
    const sin1 = std.math.sin(inv_freq1);

    try testing.expectApproxEqAbs(@as(f32, 1.0) * cos0 - @as(f32, 3.0) * sin0, head[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0) * cos1 - @as(f32, 4.0) * sin1, head[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 3.0) * cos0 + @as(f32, 1.0) * sin0, head[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0) * cos1 + @as(f32, 2.0) * sin1, head[3], 1e-6);
}

test "mrope sections rotate independent axes on configured frequency ranges" {
    const testing = std.testing;

    var table = try RoPETable.init(testing.allocator, 8, 8, 10000.0);
    defer table.deinit();

    var head = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    try applyRoPEToHeadWithPositionInPlace(
        &head,
        &table,
        decoder_types.TokenPosition.mropePosition(.{ 0, 1, 2, 0 }),
        .{ 1, 1, 2, 0 },
    );

    try testing.expectApproxEqAbs(@as(f32, 1.0), head[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), head[4], 1e-6);

    const cos1 = table.cosForPosition(1)[1];
    const sin1 = table.sinForPosition(1)[1];
    try testing.expectApproxEqAbs(@as(f32, 2.0) * cos1 - @as(f32, 6.0) * sin1, head[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0) * cos1 + @as(f32, 2.0) * sin1, head[5], 1e-6);

    const cos2_dim2 = table.cosForPosition(2)[2];
    const sin2_dim2 = table.sinForPosition(2)[2];
    try testing.expectApproxEqAbs(@as(f32, 3.0) * cos2_dim2 - @as(f32, 7.0) * sin2_dim2, head[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 7.0) * cos2_dim2 + @as(f32, 3.0) * sin2_dim2, head[6], 1e-6);

    const cos2_dim3 = table.cosForPosition(2)[3];
    const sin2_dim3 = table.sinForPosition(2)[3];
    try testing.expectApproxEqAbs(@as(f32, 4.0) * cos2_dim3 - @as(f32, 8.0) * sin2_dim3, head[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 8.0) * cos2_dim3 + @as(f32, 4.0) * sin2_dim3, head[7], 1e-6);
}
