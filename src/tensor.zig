const std = @import("std");

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    data: []f32,
    shape: [4]usize,

    pub fn init(
        allocator: std.mem.Allocator,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) !Tensor {
        const total_len = n * c * h * w;
        const data = try allocator.alloc(f32, total_len);
        return .{
            .allocator = allocator,
            .data = data,
            .shape = .{ n, c, h, w },
        };
    }

    pub fn clone(self: *const Tensor) !Tensor {
        const cloned = try Tensor.init(
            self.allocator,
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
        );
        @memcpy(cloned.data, self.data);
        return cloned;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    pub fn len(self: *const Tensor) usize {
        return self.data.len;
    }

    pub fn sameShape(self: *const Tensor, other: *const Tensor) bool {
        return std.mem.eql(usize, &self.shape, &other.shape);
    }

    pub fn index(self: *const Tensor, n: usize, c: usize, y: usize, x: usize) usize {
        const channels = self.shape[1];
        const height = self.shape[2];
        const width = self.shape[3];
        return (((n * channels) + c) * height + y) * width + x;
    }

    pub fn get(self: *const Tensor, n: usize, c: usize, y: usize, x: usize) f32 {
        return self.data[self.index(n, c, y, x)];
    }

    pub fn set(self: *Tensor, n: usize, c: usize, y: usize, x: usize, value: f32) void {
        self.data[self.index(n, c, y, x)] = value;
    }
};

test "tensor indexing is nchw" {
    const testing = std.testing;
    var tensor = try Tensor.init(testing.allocator, 1, 2, 2, 3);
    defer tensor.deinit();

    for (tensor.data, 0..) |*item, idx| item.* = @floatFromInt(idx);

    try testing.expectEqual(@as(f32, 0.0), tensor.get(0, 0, 0, 0));
    try testing.expectEqual(@as(f32, 5.0), tensor.get(0, 0, 1, 2));
    try testing.expectEqual(@as(f32, 6.0), tensor.get(0, 1, 0, 0));
    try testing.expectEqual(@as(f32, 11.0), tensor.get(0, 1, 1, 2));
}
