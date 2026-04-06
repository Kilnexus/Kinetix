const std = @import("std");

pub const DataType = enum {
    f32,
    i8,
};

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    data: []f32,
    dtype: DataType,

    pub fn initF32(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        if (shape.len == 0) return error.EmptyShape;

        var element_count: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return error.InvalidShape;
            element_count = try std.math.mul(usize, element_count, dim);
        }

        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);

        const data = try allocator.alloc(f32, element_count);
        @memset(data, 0);

        return .{
            .allocator = allocator,
            .shape = owned_shape,
            .data = data,
            .dtype = .f32,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.data);
    }

    pub fn len(self: *const Tensor) usize {
        return self.data.len;
    }
};

test "tensor initialization" {
    var tensor = try Tensor.initF32(std.testing.allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 24), tensor.len());
    try std.testing.expectEqual(@as(usize, 3), tensor.shape.len);
    try std.testing.expectEqual(DataType.f32, tensor.dtype);
}
