const std = @import("std");
const graph = @import("graph.zig");

pub const WeightsBlob = struct {
    allocator: std.mem.Allocator,
    data: []f32,

    pub fn load(allocator: std.mem.Allocator, weights_path: []const u8) !WeightsBlob {
        const cwd = std.fs.cwd();
        const stat = try cwd.statFile(weights_path);
        if (stat.size % @sizeOf(f32) != 0) return error.InvalidWeightsSize;

        const float_count = stat.size / @sizeOf(f32);
        const data = try allocator.alloc(f32, float_count);
        errdefer allocator.free(data);

        const file = try cwd.openFile(weights_path, .{});
        defer file.close();

        const bytes = std.mem.sliceAsBytes(data);
        const read_len = try file.readAll(bytes);
        if (read_len != bytes.len) return error.UnexpectedEof;
        return .{
            .allocator = allocator,
            .data = data,
        };
    }

    pub fn deinit(self: *WeightsBlob) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn slice(self: *const WeightsBlob, meta: *const graph.TensorMeta) []const f32 {
        const start = meta.offset / @sizeOf(f32);
        const len = meta.floatLen();
        return self.data[start .. start + len];
    }
};
