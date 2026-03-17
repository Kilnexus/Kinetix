const std = @import("std");

pub const ImageError = error{
    InvalidImageDimensions,
    InvalidChannelCount,
    ShapeMismatch,
};

pub const ImageU8 = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    channels: usize,
    data: []u8,

    pub fn init(
        allocator: std.mem.Allocator,
        width: usize,
        height: usize,
        channels: usize,
    ) !ImageU8 {
        if (width == 0 or height == 0) return error.InvalidImageDimensions;
        if (channels == 0) return error.InvalidChannelCount;

        return .{
            .allocator = allocator,
            .width = width,
            .height = height,
            .channels = channels,
            .data = try allocator.alloc(u8, width * height * channels),
        };
    }

    pub fn deinit(self: *ImageU8) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }

    pub fn fill(self: *ImageU8, value: u8) void {
        @memset(self.data, value);
    }

    pub fn pixelIndex(self: *const ImageU8, x: usize, y: usize, channel: usize) usize {
        return (y * self.width + x) * self.channels + channel;
    }

    pub fn get(self: *const ImageU8, x: usize, y: usize, channel: usize) u8 {
        return self.data[self.pixelIndex(x, y, channel)];
    }

    pub fn set(self: *ImageU8, x: usize, y: usize, channel: usize, value: u8) void {
        self.data[self.pixelIndex(x, y, channel)] = value;
    }
};
