const std = @import("std");

pub const Config = struct {
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: []const u8,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    torch_dtype: []const u8 = "float32",
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: Config,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
    }
};

pub fn loadFromFile(backing_allocator: std.mem.Allocator, path: []const u8) !ParsedConfig {
    var arena = std.heap.ArenaAllocator.init(backing_allocator);
    errdefer arena.deinit();

    const allocator = arena.allocator();
    const bytes = try readFileAllocAtPath(allocator, path, 1024 * 1024);
    const config = try std.json.parseFromSliceLeaky(Config, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{
        .arena = arena,
        .value = config,
    };
}

fn readFileAllocAtPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    if (std.fs.path.isAbsolute(path)) {
        const file = try std.fs.openFileAbsolute(path, .{});
        defer file.close();
        return file.readToEndAlloc(allocator, max_bytes);
    }
    return std.fs.cwd().readFileAlloc(allocator, path, max_bytes);
}
