const std = @import("std");
const bfloat16 = @import("../../common/bfloat16.zig");
const q8 = @import("index.zig");
const common = @import("common.zig");

const handwritten_q8_head_dim = common.handwritten_q8_head_dim;
const handwritten_q8_scale_groups = common.handwritten_q8_scale_groups;

test "q8 attention handwritten 128 full path matches generic path" {
    const testing = std.testing;

    const seq_len = 5;
    const num_query_heads = 4;
    const num_key_value_heads = 2;
    const head_dim = handwritten_q8_head_dim;
    const total_query = num_query_heads * head_dim;
    const total_cache = seq_len * num_key_value_heads * head_dim;
    const total_scales = seq_len * num_key_value_heads * handwritten_q8_scale_groups;

    var query: [total_query]f32 = undefined;
    var key_cache: [total_cache]i8 = undefined;
    var value_cache: [total_cache]i8 = undefined;
    var key_scales: [total_scales]u16 = undefined;
    var value_scales: [total_scales]u16 = undefined;
    var scores_generic: [seq_len]f32 = undefined;
    var scores_handwritten: [seq_len]f32 = undefined;
    var output_generic: [total_query]f32 = undefined;
    var output_handwritten: [total_query]f32 = undefined;

    for (&query, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 37)) - 18)) / 9.0;
    }
    for (&key_cache, 0..) |*value, idx| {
        value.* = @intCast(@as(i16, @intCast((idx * 11 + 5) % 255)) - 127);
        value_cache[idx] = @intCast(@as(i16, @intCast((idx * 13 + 9) % 255)) - 127);
    }
    for (&key_scales, 0..) |*value, idx| {
        value.* = bfloat16.fromF32(@as(f32, @floatFromInt((idx % handwritten_q8_scale_groups) + 1)) / 127.0);
        value_scales[idx] = bfloat16.fromF32(@as(f32, @floatFromInt((idx % handwritten_q8_scale_groups) + 2)) / 127.0);
    }

    try q8.testingScaledDotProductAttentionSingleQueryQ8CacheGeneric(
        &output_generic,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        handwritten_q8_scale_groups,
        &scores_generic,
    );

    try q8.testingScaledDotProductAttentionSingleQueryQ8Cache128(
        &output_handwritten,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        &scores_handwritten,
    );

    for (output_generic, output_handwritten) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "q8 attention head-major path matches token-major path" {
    const testing = std.testing;

    const seq_len = 5;
    const num_query_heads = 4;
    const num_key_value_heads = 2;
    const head_dim = handwritten_q8_head_dim;
    const scale_groups_per_head = handwritten_q8_scale_groups;
    const total_query = num_query_heads * head_dim;
    const total_cache = seq_len * num_key_value_heads * head_dim;
    const total_scales = seq_len * num_key_value_heads * scale_groups_per_head;
    const head_data_stride = seq_len * head_dim;
    const head_scale_stride = seq_len * scale_groups_per_head;

    var query: [total_query]f32 = undefined;
    var key_cache_token_major: [total_cache]i8 = undefined;
    var value_cache_token_major: [total_cache]i8 = undefined;
    var key_scales_token_major: [total_scales]u16 = undefined;
    var value_scales_token_major: [total_scales]u16 = undefined;
    var key_cache_head_major: [total_cache]i8 = undefined;
    var value_cache_head_major: [total_cache]i8 = undefined;
    var key_scales_head_major: [total_scales]u16 = undefined;
    var value_scales_head_major: [total_scales]u16 = undefined;
    var scores_token_major: [seq_len]f32 = undefined;
    var scores_head_major: [seq_len]f32 = undefined;
    var output_token_major: [total_query]f32 = undefined;
    var output_head_major: [total_query]f32 = undefined;

    for (&query, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 37)) - 18)) / 9.0;
    }
    for (&key_cache_token_major, 0..) |*value, idx| {
        value.* = @intCast(@as(i16, @intCast((idx * 11 + 5) % 255)) - 127);
        value_cache_token_major[idx] = @intCast(@as(i16, @intCast((idx * 13 + 9) % 255)) - 127);
    }
    for (&key_scales_token_major, 0..) |*value, idx| {
        value.* = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 1)) / 127.0);
        value_scales_token_major[idx] = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 2)) / 127.0);
    }

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * scale_groups_per_head;
            const head_major_data_start = head_idx * head_data_stride + pos * head_dim;
            const head_major_scale_start = head_idx * head_scale_stride + pos * scale_groups_per_head;
            @memcpy(key_cache_head_major[head_major_data_start .. head_major_data_start + head_dim], key_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(value_cache_head_major[head_major_data_start .. head_major_data_start + head_dim], value_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(key_scales_head_major[head_major_scale_start .. head_major_scale_start + scale_groups_per_head], key_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
            @memcpy(value_scales_head_major[head_major_scale_start .. head_major_scale_start + scale_groups_per_head], value_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
        }
    }

    try q8.scaledDotProductAttentionSingleQueryQ8Cache(
        &output_token_major,
        &query,
        &key_cache_token_major,
        &key_scales_token_major,
        &value_cache_token_major,
        &value_scales_token_major,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_token_major,
    );

    try q8.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
        &output_head_major,
        &query,
        &key_cache_head_major,
        &key_scales_head_major,
        &value_cache_head_major,
        &value_scales_head_major,
        head_data_stride,
        head_scale_stride,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_head_major,
    );

    for (output_token_major, output_head_major) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "q8 attention paged head-major path matches head-major path" {
    const testing = std.testing;

    const seq_len = 37;
    const page_len = 32;
    const pages_per_head = 2;
    const num_query_heads = 4;
    const num_key_value_heads = 2;
    const head_dim = handwritten_q8_head_dim;
    const scale_groups_per_head = handwritten_q8_scale_groups;
    const total_query = num_query_heads * head_dim;
    const total_cache = seq_len * num_key_value_heads * head_dim;
    const total_scales = seq_len * num_key_value_heads * scale_groups_per_head;
    const head_data_stride = seq_len * head_dim;
    const head_scale_stride = seq_len * scale_groups_per_head;
    const page_data_stride = page_len * head_dim;
    const page_scale_stride = page_len * scale_groups_per_head;
    const paged_total_cache = num_key_value_heads * pages_per_head * page_data_stride;
    const paged_total_scales = num_key_value_heads * pages_per_head * page_scale_stride;

    var query: [total_query]f32 = undefined;
    var key_cache_token_major: [total_cache]i8 = undefined;
    var value_cache_token_major: [total_cache]i8 = undefined;
    var key_scales_token_major: [total_scales]u16 = undefined;
    var value_scales_token_major: [total_scales]u16 = undefined;
    var key_cache_head_major: [total_cache]i8 = undefined;
    var value_cache_head_major: [total_cache]i8 = undefined;
    var key_scales_head_major: [total_scales]u16 = undefined;
    var value_scales_head_major: [total_scales]u16 = undefined;
    var key_cache_paged: [paged_total_cache]i8 = [_]i8{0} ** paged_total_cache;
    var value_cache_paged: [paged_total_cache]i8 = [_]i8{0} ** paged_total_cache;
    var key_scales_paged: [paged_total_scales]u16 = [_]u16{0} ** paged_total_scales;
    var value_scales_paged: [paged_total_scales]u16 = [_]u16{0} ** paged_total_scales;
    var scores_head_major: [seq_len]f32 = undefined;
    var scores_paged: [seq_len]f32 = undefined;
    var output_head_major: [total_query]f32 = undefined;
    var output_paged: [total_query]f32 = undefined;

    for (&query, 0..) |*value, idx| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 37)) - 18)) / 9.0;
    }
    for (&key_cache_token_major, 0..) |*value, idx| {
        value.* = @intCast(@as(i16, @intCast((idx * 11 + 5) % 255)) - 127);
        value_cache_token_major[idx] = @intCast(@as(i16, @intCast((idx * 13 + 9) % 255)) - 127);
    }
    for (&key_scales_token_major, 0..) |*value, idx| {
        value.* = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 1)) / 127.0);
        value_scales_token_major[idx] = bfloat16.fromF32(@as(f32, @floatFromInt((idx % scale_groups_per_head) + 2)) / 127.0);
    }

    for (0..num_key_value_heads) |head_idx| {
        for (0..seq_len) |pos| {
            const token_major_data_start = (pos * num_key_value_heads + head_idx) * head_dim;
            const token_major_scale_start = (pos * num_key_value_heads + head_idx) * scale_groups_per_head;
            const linear_data_start = head_idx * head_data_stride + pos * head_dim;
            const linear_scale_start = head_idx * head_scale_stride + pos * scale_groups_per_head;
            const page_idx = pos / page_len;
            const page_offset = pos % page_len;
            const paged_data_start = head_idx * pages_per_head * page_data_stride + page_idx * page_data_stride + page_offset * head_dim;
            const paged_scale_start = head_idx * pages_per_head * page_scale_stride + page_idx * page_scale_stride + page_offset * scale_groups_per_head;
            @memcpy(key_cache_head_major[linear_data_start .. linear_data_start + head_dim], key_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(value_cache_head_major[linear_data_start .. linear_data_start + head_dim], value_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(key_scales_head_major[linear_scale_start .. linear_scale_start + scale_groups_per_head], key_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
            @memcpy(value_scales_head_major[linear_scale_start .. linear_scale_start + scale_groups_per_head], value_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
            @memcpy(key_cache_paged[paged_data_start .. paged_data_start + head_dim], key_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(value_cache_paged[paged_data_start .. paged_data_start + head_dim], value_cache_token_major[token_major_data_start .. token_major_data_start + head_dim]);
            @memcpy(key_scales_paged[paged_scale_start .. paged_scale_start + scale_groups_per_head], key_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
            @memcpy(value_scales_paged[paged_scale_start .. paged_scale_start + scale_groups_per_head], value_scales_token_major[token_major_scale_start .. token_major_scale_start + scale_groups_per_head]);
        }
    }

    try q8.scaledDotProductAttentionSingleQueryQ8CacheHeadMajor(
        &output_head_major,
        &query,
        &key_cache_head_major,
        &key_scales_head_major,
        &value_cache_head_major,
        &value_scales_head_major,
        head_data_stride,
        head_scale_stride,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_head_major,
    );

    try q8.scaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor(
        &output_paged,
        &query,
        &key_cache_paged,
        &key_scales_paged,
        &value_cache_paged,
        &value_scales_paged,
        pages_per_head * page_data_stride,
        pages_per_head * page_scale_stride,
        page_data_stride,
        page_scale_stride,
        page_len,
        pages_per_head,
        seq_len,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        &scores_paged,
    );

    for (output_head_major, output_paged) |expected, actual| {
        try testing.expectApproxEqAbs(expected, actual, 1e-5);
    }
}

test "single-query attention supports q8 kv cache" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]i8{ 127, 0, 0, 127 };
    const value_cache = [_]i8{ 64, 6, 127, 13 };
    const key_scales = [_]u16{ bfloat16.fromF32(1.0 / 127.0), bfloat16.fromF32(1.0 / 127.0) };
    const value_scales = [_]u16{ bfloat16.fromF32(20.0 / 127.0), bfloat16.fromF32(20.0 / 127.0) };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try q8.scaledDotProductAttentionSingleQueryQ8Cache(
        &output,
        &query,
        &key_cache,
        &key_scales,
        &value_cache,
        &value_scales,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 9.0);
    try testing.expect(output[0] < 20.5);
    try testing.expect(output[1] > 0.5);
    try testing.expect(output[1] < 2.5);
}
