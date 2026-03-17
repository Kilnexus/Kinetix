const std = @import("std");

pub const TensorDesc = struct {
    shape: [4]usize,
    len: usize,
};

pub fn shapeLen(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| total *= dim;
    return total;
}

pub fn printRoadmap(writer: anytype) !void {
    try writer.writeAll(
        \\Full runtime status:
        \\1. Graph and weights export: ready
        \\2. Zig graph loader: ready
        \\3. Primitive tensor ops: implemented
        \\4. Composite YOLO11s blocks: pending
        \\5. Detect + DFL + NMS: pending
        \\6. End-to-end parity check: pending
        \\
    );
}
