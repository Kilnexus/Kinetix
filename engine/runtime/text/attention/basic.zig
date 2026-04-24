const std = @import("std");
const shared_basic = @import("shared_ops").kernels.attention.basic;

pub const softmaxInPlace = shared_basic.softmaxInPlace;
pub const scaledDotProductAttentionSingleQuery = shared_basic.scaledDotProductAttentionSingleQuery;
pub const scaledDotProductAttentionSingleQueryBf16Cache = shared_basic.scaledDotProductAttentionSingleQueryBf16Cache;

inline fn bf16FromF32(value: f32) u16 {
    const raw: u32 = @bitCast(value);
    const lsb = (raw >> 16) & 1;
    const rounding_bias: u32 = 0x7fff + lsb;
    return @truncate((raw + rounding_bias) >> 16);
}

test "softmax normalizes values" {
    const testing = std.testing;

    var values = [_]f32{ 1.0, 2.0, 3.0 };
    try softmaxInPlace(&values);

    const sum = values[0] + values[1] + values[2];
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try testing.expect(values[2] > values[1]);
    try testing.expect(values[1] > values[0]);
}

test "single-query attention attends over one kv head" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const value_cache = [_]f32{
        10.0, 1.0,
        20.0, 2.0,
    };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 10.0);
    try testing.expect(output[0] < 20.0);
    try testing.expect(output[1] > 1.0);
    try testing.expect(output[1] < 2.0);
}

test "single-query attention supports grouped query attention" {
    const testing = std.testing;

    const query = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const key_cache = [_]f32{
        1.0, 0.0,
    };
    const value_cache = [_]f32{
        5.0, 6.0,
    };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    var scores = [_]f32{0.0};

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        1,
        2,
        1,
        2,
        &scores,
    );

    try testing.expectApproxEqAbs(@as(f32, 5.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 5.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), output[3], 1e-6);
}

test "single-query attention maps grouped query heads to different kv heads" {
    const testing = std.testing;

    const query = [_]f32{
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
    };
    const key_cache = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    const value_cache = [_]f32{
        11.0, 12.0,
        21.0, 22.0,
    };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var scores = [_]f32{0.0};

    try scaledDotProductAttentionSingleQuery(
        &output,
        &query,
        &key_cache,
        &value_cache,
        1,
        4,
        2,
        2,
        &scores,
    );

    try testing.expectApproxEqAbs(@as(f32, 11.0), output[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 12.0), output[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 11.0), output[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 12.0), output[3], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 21.0), output[4], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 22.0), output[5], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 21.0), output[6], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 22.0), output[7], 1e-6);
}

test "single-query attention supports bf16 kv cache" {
    const testing = std.testing;

    const query = [_]f32{ 1.0, 0.0 };
    const key_cache = [_]u16{
        bf16FromF32(1.0), bf16FromF32(0.0),
        bf16FromF32(0.0), bf16FromF32(1.0),
    };
    const value_cache = [_]u16{
        bf16FromF32(10.0), bf16FromF32(1.0),
        bf16FromF32(20.0), bf16FromF32(2.0),
    };
    var output = [_]f32{ 0.0, 0.0 };
    var scores = [_]f32{ 0.0, 0.0 };

    try scaledDotProductAttentionSingleQueryBf16Cache(
        &output,
        &query,
        &key_cache,
        &value_cache,
        2,
        1,
        1,
        2,
        &scores,
    );

    try testing.expect(output[0] > 10.0);
    try testing.expect(output[0] < 20.0);
    try testing.expect(output[1] > 1.0);
    try testing.expect(output[1] < 2.0);
}
