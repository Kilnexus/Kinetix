const std = @import("std");

pub const DType = enum {
    i32,
    f32,
};

pub const Buffer = union(DType) {
    i32: []i32,
    f32: []f32,
};

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    buffer: Buffer,

    pub fn fromF32(allocator: std.mem.Allocator, shape: []const usize, values: []const f32) !Tensor {
        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);
        const owned_values = try allocator.dupe(f32, values);
        errdefer allocator.free(owned_values);
        const tensor = Tensor{ .allocator = allocator, .shape = owned_shape, .buffer = .{ .f32 = owned_values } };
        if (tensor.elementCount() != values.len) return error.TensorElementCountMismatch;
        return tensor;
    }

    pub fn fromI32(allocator: std.mem.Allocator, shape: []const usize, values: []const i32) !Tensor {
        const owned_shape = try allocator.dupe(usize, shape);
        errdefer allocator.free(owned_shape);
        const owned_values = try allocator.dupe(i32, values);
        errdefer allocator.free(owned_values);
        const tensor = Tensor{ .allocator = allocator, .shape = owned_shape, .buffer = .{ .i32 = owned_values } };
        if (tensor.elementCount() != values.len) return error.TensorElementCountMismatch;
        return tensor;
    }

    pub fn clone(self: Tensor, allocator: std.mem.Allocator) !Tensor {
        return switch (self.buffer) {
            .f32 => |values| try fromF32(allocator, self.shape, values),
            .i32 => |values| try fromI32(allocator, self.shape, values),
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.shape);
        switch (self.buffer) {
            .f32 => |values| self.allocator.free(values),
            .i32 => |values| self.allocator.free(values),
        }
        self.* = undefined;
    }

    pub fn dtype(self: Tensor) DType {
        return std.meta.activeTag(self.buffer);
    }

    pub fn elementCount(self: Tensor) usize {
        if (self.shape.len == 0) return 0;
        var total: usize = 1;
        for (self.shape) |dim| total *= dim;
        return total;
    }

    pub fn sameShape(self: Tensor, other: Tensor) bool {
        return std.mem.eql(usize, self.shape, other.shape);
    }
};

test "runtime tensor owns shape and values" {
    var tensor = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer tensor.deinit();
    try std.testing.expectEqual(@as(usize, 4), tensor.elementCount());
    try std.testing.expectEqual(DType.f32, tensor.dtype());
}
