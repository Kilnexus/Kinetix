const std = @import("std");
const MemoryPool = @import("../core/memory_pool.zig").MemoryPool;
const Model = @import("../io/model.zig").Model;
const Image = @import("../io/image.zig").Image;

pub const InferRequest = struct {
    model_path: []const u8,
    image_path: []const u8,
};

pub const InferResult = struct {
    loaded_tensors: usize,
    image_width: usize,
    image_height: usize,
};

pub const Pipeline = struct {
    allocator: std.mem.Allocator,
    pool: MemoryPool,

    pub fn init(allocator: std.mem.Allocator) Pipeline {
        return .{
            .allocator = allocator,
            .pool = MemoryPool.init(allocator),
        };
    }

    pub fn deinit(self: *Pipeline) void {
        self.pool.deinit();
    }

    pub fn infer(self: *Pipeline, req: InferRequest) !InferResult {
        self.pool.reset();

        var model = try Model.loadFromFile(self.allocator, req.model_path);
        defer model.deinit();

        var image = try Image.loadPpmFile(self.allocator, req.image_path);
        defer image.deinit();

        // Placeholder execution result. The next milestone will map model graph to op kernels.
        return .{
            .loaded_tensors = model.tensorCount(),
            .image_width = image.width,
            .image_height = image.height,
        };
    }
};

test "pipeline basic infer skeleton" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    // Write minimal model file.
    var model_file = try tmp.dir.createFile("m.swm", .{});
    defer model_file.close();
    var mw_impl = model_file.writer(&.{});
    const mw = &mw_impl.interface;
    try mw.writeAll(&[_]u8{ 'S', 'W', 'O', 'C', 'R', '0', '1', 0 });
    try mw.writeInt(u32, 0, .little);
    try mw.flush();

    // Write 1x1 rgb PPM.
    var image_file = try tmp.dir.createFile("i.ppm", .{});
    defer image_file.close();
    var iw_impl = image_file.writer(&.{});
    const iw = &iw_impl.interface;
    try iw.writeAll("P6\n1 1\n255\n");
    try iw.writeAll(&[_]u8{ 1, 2, 3 });
    try iw.flush();

    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa_state.deinit() == .ok);
    const gpa = gpa_state.allocator();

    var pipeline = Pipeline.init(gpa);
    defer pipeline.deinit();

    // Use cwd-relative workaround for this test by opening tmp dir and changing path lookup.
    // We call IO loaders directly from tmp dir in this test to validate data first.
    var model = try Model.loadFromDir(gpa, tmp.dir, "m.swm");
    defer model.deinit();
    var img = try Image.loadPpmFromDir(gpa, tmp.dir, "i.ppm");
    defer img.deinit();

    try std.testing.expectEqual(@as(usize, 0), model.tensorCount());
    try std.testing.expectEqual(@as(usize, 1), img.width);
    try std.testing.expectEqual(@as(usize, 1), img.height);
}
