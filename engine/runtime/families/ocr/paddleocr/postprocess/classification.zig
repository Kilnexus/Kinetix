const std = @import("std");
const Tensor = @import("shared_graph").runtime.tensor.Tensor;

pub const Orientation = enum {
    normal,
    rotate_180,
    unknown,
};

pub const Result = struct {
    orientation: Orientation,
    class_index: usize,
    confidence: f32,
};

pub fn classifyOrientation(tensor: Tensor) !Result {
    if (tensor.buffer != .f32) return error.UnsupportedTensorDType;
    const values = tensor.buffer.f32;
    if (values.len == 0) return error.TensorElementCountMismatch;

    const class_count = switch (tensor.shape.len) {
        1 => tensor.shape[0],
        2 => blk: {
            if (tensor.shape[0] != 1) return error.UnsupportedTensorShape;
            break :blk tensor.shape[1];
        },
        else => return error.UnsupportedTensorRank,
    };
    if (class_count == 0 or values.len < class_count) return error.TensorElementCountMismatch;

    var best_index: usize = 0;
    var best_value = values[0];
    for (values[1..class_count], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }

    return .{
        .orientation = switch (best_index) {
            0 => .normal,
            1 => .rotate_180,
            else => .unknown,
        },
        .class_index = best_index,
        .confidence = best_value,
    };
}

test "paddleocr classifier maps two class output to orientation" {
    var tensor = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2 }, &.{ 0.1, 0.9 });
    defer tensor.deinit();

    const result = try classifyOrientation(tensor);
    try std.testing.expectEqual(Orientation.rotate_180, result.orientation);
    try std.testing.expectEqual(@as(usize, 1), result.class_index);
}

