const std = @import("std");
const bundle = @import("../../bundle/index.zig");
const shared_graph = @import("../../../../../shared/graph/index.zig");

const io = std.Options.debug_io;

pub const Summary = struct {
    loaded_graph_count: usize = 0,
    tts_prefill_node_count: usize = 0,
    tts_decode_step_node_count: usize = 0,
    codec_decode_full_node_count: usize = 0,
    total_initializer_count: usize = 0,
    external_initializer_count: usize = 0,
};

pub fn inspect(
    allocator: std.mem.Allocator,
    tts_meta_path: []const u8,
    codec_meta_path: []const u8,
    tts: bundle.TtsSummary,
    codec: bundle.CodecConfig,
) !Summary {
    var summary = Summary{};
    const tts_dir = std.fs.path.dirname(tts_meta_path) orelse ".";
    const codec_dir = std.fs.path.dirname(codec_meta_path) orelse ".";

    if (try loadOptional(allocator, tts_dir, tts.files.prefill)) |metadata_value| {
        var metadata = metadata_value;
        defer metadata.deinit();
        summary.loaded_graph_count += 1;
        summary.tts_prefill_node_count = metadata.graph.node_count;
        addInitializers(&summary, &metadata);
    }

    if (try loadOptional(allocator, tts_dir, tts.files.decode_step)) |metadata_value| {
        var metadata = metadata_value;
        defer metadata.deinit();
        summary.loaded_graph_count += 1;
        summary.tts_decode_step_node_count = metadata.graph.node_count;
        addInitializers(&summary, &metadata);
    }

    if (try loadOptional(allocator, codec_dir, codec.files.decode_full)) |metadata_value| {
        var metadata = metadata_value;
        defer metadata.deinit();
        summary.loaded_graph_count += 1;
        summary.codec_decode_full_node_count = metadata.graph.node_count;
        addInitializers(&summary, &metadata);
    }

    return summary;
}

fn loadOptional(
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    file_name: []const u8,
) !?shared_graph.onnx.metadata.ModelMetadata {
    if (file_name.len == 0) return null;
    const full_path = try std.fs.path.join(allocator, &.{ base_dir, file_name });
    defer allocator.free(full_path);
    if (!pathExists(full_path)) return null;
    return try shared_graph.onnx.metadata.loadFromFile(allocator, full_path);
}

fn addInitializers(summary: *Summary, metadata: *const shared_graph.onnx.metadata.ModelMetadata) void {
    summary.total_initializer_count += metadata.graph.initializers.len;
    for (metadata.graph.initializers) |initializer| {
        if (initializer.isExternal()) summary.external_initializer_count += 1;
    }
}

fn pathExists(path: []const u8) bool {
    const file = if (std.fs.path.isAbsolute(path))
        std.Io.Dir.openFileAbsolute(io, path, .{})
    else
        std.Io.Dir.cwd().openFile(io, path, .{});
    if (file) |handle| {
        handle.close(io);
        return true;
    } else |_| {
        return false;
    }
}

test "moss onnx metadata planning tolerates missing graph files" {
    const summary = try inspect(std.testing.allocator, "missing/tts_meta.json", "missing/codec_meta.json", .{}, .{});
    try std.testing.expectEqual(@as(usize, 0), summary.loaded_graph_count);
    try std.testing.expectEqual(@as(usize, 0), summary.total_initializer_count);
}
