const std = @import("std");
const ArenaPool = @import("../core/memory/arena_pool.zig").ArenaPool;
const ocr_artifacts = @import("../artifacts/ocr/ocr.zig");

pub const InferRequest = struct {
    model_path: []const u8,
    image_path: []const u8,
};

pub const InferResult = struct {
    loaded_tensors: usize,
    image_width: usize,
    image_height: usize,
};

pub const OCRPipeline = struct {
    allocator: std.mem.Allocator,
    pool: ArenaPool,

    pub fn init(allocator: std.mem.Allocator) OCRPipeline {
        return .{
            .allocator = allocator,
            .pool = ArenaPool.init(allocator),
        };
    }

    pub fn deinit(self: *OCRPipeline) void {
        self.pool.deinit();
    }

    pub fn infer(self: *OCRPipeline, req: InferRequest) !InferResult {
        self.pool.reset();

        var model = try ocr_artifacts.Model.loadFromFile(self.allocator, req.model_path);
        defer model.deinit();

        var image = try ocr_artifacts.Image.loadPpmFile(self.allocator, req.image_path);
        defer image.deinit();

        return .{
            .loaded_tensors = model.tensorCount(),
            .image_width = image.width,
            .image_height = image.height,
        };
    }
};

test "ocr pipeline executes shared infer skeleton" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var model_file = try tmp.dir.createFile("demo.swm", .{});
    defer model_file.close();
    var model_writer_impl = model_file.writer(&.{});
    const model_writer = &model_writer_impl.interface;
    try model_writer.writeAll(&[_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 });
    try model_writer.writeInt(u32, 0, .little);
    try model_writer.flush();

    var image_file = try tmp.dir.createFile("demo.ppm", .{});
    defer image_file.close();
    var image_writer_impl = image_file.writer(&.{});
    const image_writer = &image_writer_impl.interface;
    try image_writer.writeAll("P6\n1 1\n255\n");
    try image_writer.writeAll(&[_]u8{ 1, 2, 3 });
    try image_writer.flush();

    const model_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.swm");
    defer std.testing.allocator.free(model_path);
    const image_path = try tmp.dir.realpathAlloc(std.testing.allocator, "demo.ppm");
    defer std.testing.allocator.free(image_path);

    var pipeline = OCRPipeline.init(std.testing.allocator);
    defer pipeline.deinit();

    const result = try pipeline.infer(.{
        .model_path = model_path,
        .image_path = image_path,
    });
    try std.testing.expectEqual(@as(usize, 0), result.loaded_tensors);
    try std.testing.expectEqual(@as(usize, 1), result.image_width);
    try std.testing.expectEqual(@as(usize, 1), result.image_height);
}
