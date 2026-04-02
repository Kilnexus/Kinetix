const std = @import("std");

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: []const u8,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    torch_dtype: []const u8 = "float32",

    pub fn firstArchitecture(self: Config) ?[]const u8 {
        if (self.architectures.len == 0) return null;
        return self.architectures[0];
    }
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
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    const config = try std.json.parseFromSliceLeaky(Config, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });

    return .{
        .arena = arena,
        .value = config,
    };
}
