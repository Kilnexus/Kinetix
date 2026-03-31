const std = @import("std");

pub fn matmul(
    a: []const f32,
    b: []const f32,
    out: []f32,
    m: usize,
    k: usize,
    n: usize,
) !void {
    if (a.len != m * k) return error.ShapeMismatch;
    if (b.len != k * n) return error.ShapeMismatch;
    if (out.len != m * n) return error.ShapeMismatch;

    @memset(out, 0);
    for (0..m) |row| {
        for (0..k) |mid| {
            const a_val = a[row * k + mid];
            for (0..n) |col| {
                out[row * n + col] += a_val * b[mid * n + col];
            }
        }
    }
}

test "matmul 2x3 and 3x2" {
    const a = [_]f32{
        1, 2, 3,
        4, 5, 6,
    };
    const b = [_]f32{
        7,  8,
        9,  10,
        11, 12,
    };
    var out = [_]f32{ 0, 0, 0, 0 };
    try matmul(&a, &b, &out, 2, 3, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 58), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 64), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 139), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 154), out[3], 1e-6);
}
