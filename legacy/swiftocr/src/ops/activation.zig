const std = @import("std");

pub fn relu(input: []const f32, output: []f32) !void {
    if (input.len != output.len) return error.ShapeMismatch;
    for (input, 0..) |value, i| {
        output[i] = if (value > 0.0) value else 0.0;
    }
}

pub fn hardSwish(input: []const f32, output: []f32) !void {
    if (input.len != output.len) return error.ShapeMismatch;
    for (input, 0..) |value, i| {
        const gate = std.math.clamp(value + 3.0, 0.0, 6.0) / 6.0;
        output[i] = value * gate;
    }
}

test "relu activation" {
    const in = [_]f32{ -1.5, 0.0, 2.25 };
    var out = [_]f32{ 0.0, 0.0, 0.0 };
    try relu(&in, &out);
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    try std.testing.expectEqual(@as(f32, 0.0), out[1]);
    try std.testing.expectEqual(@as(f32, 2.25), out[2]);
}
