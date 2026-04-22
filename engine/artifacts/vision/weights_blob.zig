const std = @import("std");
const graph = @import("graph");
const io = std.Options.debug_io;

pub const WeightsBlob = struct {
    allocator: std.mem.Allocator,
    data: []f32,

    pub fn load(allocator: std.mem.Allocator, weights_path: []const u8) !WeightsBlob {
        var file = if (std.fs.path.isAbsolute(weights_path))
            try std.Io.Dir.openFileAbsolute(io, weights_path, .{})
        else
            try std.Io.Dir.cwd().openFile(io, weights_path, .{});
        defer file.close(io);

        const stat = try file.stat(io);
        if (stat.size % @sizeOf(f32) != 0) return error.InvalidWeightsSize;

        const float_count: usize = @intCast(stat.size / @sizeOf(f32));
        const data = try allocator.alloc(f32, float_count);
        errdefer allocator.free(data);

        const bytes = std.mem.sliceAsBytes(data);
        const read_len = try file.readPositionalAll(io, bytes, 0);
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
