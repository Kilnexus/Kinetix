const std = @import("std");
const bundle = @import("../../bundle/index.zig");
const shared_graph = @import("../../../../../shared/graph/index.zig");

const io = std.Options.debug_io;

pub const max_unsupported_op_entries: usize = 24;

pub const UnsupportedOpEntry = struct {
    op_type: []const u8 = "",
    count: usize = 0,
};

pub const Summary = struct {
    loaded_graph_count: usize = 0,
    tts_prefill_node_count: usize = 0,
    tts_decode_step_node_count: usize = 0,
    codec_decode_full_node_count: usize = 0,
    total_initializer_count: usize = 0,
    external_initializer_count: usize = 0,
    supported_node_count: usize = 0,
    unsupported_node_count: usize = 0,
    unsupported_ops: [max_unsupported_op_entries]UnsupportedOpEntry = [_]UnsupportedOpEntry{.{}} ** max_unsupported_op_entries,
    unsupported_op_entry_count: usize = 0,
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
        addOperatorSupport(&summary, &metadata, "tts_prefill");
    }

    if (try loadOptional(allocator, tts_dir, tts.files.decode_step)) |metadata_value| {
        var metadata = metadata_value;
        defer metadata.deinit();
        summary.loaded_graph_count += 1;
        summary.tts_decode_step_node_count = metadata.graph.node_count;
        addInitializers(&summary, &metadata);
        addOperatorSupport(&summary, &metadata, "tts_decode_step");
    }

    if (try loadOptional(allocator, codec_dir, codec.files.decode_full)) |metadata_value| {
        var metadata = metadata_value;
        defer metadata.deinit();
        summary.loaded_graph_count += 1;
        summary.codec_decode_full_node_count = metadata.graph.node_count;
        addInitializers(&summary, &metadata);
        addOperatorSupport(&summary, &metadata, "codec_decode_full");
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

fn addOperatorSupport(summary: *Summary, metadata: *const shared_graph.onnx.metadata.ModelMetadata, stage: []const u8) void {
    for (metadata.graph.nodes) |node| {
        if (shared_graph.runtime.ops.isSupported(node.op_type)) {
            summary.supported_node_count += 1;
        } else {
            summary.unsupported_node_count += 1;
            addUnsupportedOp(summary, stage, node.op_type);
        }
    }
}

fn addUnsupportedOp(summary: *Summary, stage: []const u8, op_type: []const u8) void {
    var key_buffer: [96]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buffer, "{s}:{s}", .{ stage, op_type }) catch op_type;
    for (summary.unsupported_ops[0..summary.unsupported_op_entry_count]) |*entry| {
        if (std.mem.eql(u8, entry.op_type, key)) {
            entry.count += 1;
            sortUnsupportedOps(summary);
            return;
        }
    }
    if (summary.unsupported_op_entry_count < summary.unsupported_ops.len) {
        const index = summary.unsupported_op_entry_count;
        summary.unsupported_ops[index] = .{
            .op_type = internUnsupportedOpKey(key),
            .count = 1,
        };
        summary.unsupported_op_entry_count += 1;
        sortUnsupportedOps(summary);
        return;
    }
    if (summary.unsupported_ops[summary.unsupported_ops.len - 1].count < 1) {
        summary.unsupported_ops[summary.unsupported_ops.len - 1] = .{
            .op_type = internUnsupportedOpKey(key),
            .count = 1,
        };
        sortUnsupportedOps(summary);
    }
}

fn internUnsupportedOpKey(key: []const u8) []const u8 {
    inline for (known_unsupported_op_keys) |known| {
        if (std.mem.eql(u8, key, known)) return known;
    }
    return "other";
}

fn sortUnsupportedOps(summary: *Summary) void {
    std.mem.sort(
        UnsupportedOpEntry,
        summary.unsupported_ops[0..summary.unsupported_op_entry_count],
        {},
        struct {
            fn lessThan(_: void, lhs: UnsupportedOpEntry, rhs: UnsupportedOpEntry) bool {
                return lhs.count > rhs.count;
            }
        }.lessThan,
    );
}

const known_unsupported_op_keys = [_][]const u8{
    "tts_prefill:Constant",
    "tts_prefill:Cast",
    "tts_prefill:Reshape",
    "tts_prefill:Transpose",
    "tts_prefill:Gather",
    "tts_prefill:Concat",
    "tts_prefill:Unsqueeze",
    "tts_prefill:Squeeze",
    "tts_prefill:Shape",
    "tts_prefill:Slice",
    "tts_prefill:Softmax",
    "tts_prefill:LayerNormalization",
    "tts_prefill:Gemm",
    "tts_decode_step:Constant",
    "tts_decode_step:Cast",
    "tts_decode_step:Reshape",
    "tts_decode_step:Transpose",
    "tts_decode_step:Gather",
    "tts_decode_step:Concat",
    "tts_decode_step:Unsqueeze",
    "tts_decode_step:Squeeze",
    "tts_decode_step:Shape",
    "tts_decode_step:Slice",
    "tts_decode_step:Softmax",
    "tts_decode_step:LayerNormalization",
    "tts_decode_step:Gemm",
    "codec_decode_full:Constant",
    "codec_decode_full:Cast",
    "codec_decode_full:Reshape",
    "codec_decode_full:Transpose",
    "codec_decode_full:Gather",
    "codec_decode_full:Concat",
    "codec_decode_full:Unsqueeze",
    "codec_decode_full:Squeeze",
    "codec_decode_full:Shape",
    "codec_decode_full:Slice",
    "codec_decode_full:Conv",
    "codec_decode_full:ConvTranspose",
    "codec_decode_full:LeakyRelu",
};

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

test "moss onnx metadata planning aggregates unsupported ops" {
    var summary = Summary{};
    var metadata = try shared_graph.onnx.metadata.parseModel(std.testing.allocator, try tinyUnsupportedModel(std.testing.allocator));
    defer metadata.deinit();
    addOperatorSupport(&summary, &metadata, "tts_prefill");

    try std.testing.expectEqual(@as(usize, 1), summary.supported_node_count);
    try std.testing.expectEqual(@as(usize, 2), summary.unsupported_node_count);
    try std.testing.expectEqual(@as(usize, 1), summary.unsupported_op_entry_count);
    try std.testing.expectEqualStrings("tts_prefill:Reshape", summary.unsupported_ops[0].op_type);
    try std.testing.expectEqual(@as(usize, 2), summary.unsupported_ops[0].count);
}

fn tinyUnsupportedModel(allocator: std.mem.Allocator) ![]u8 {
    var graph = std.ArrayList(u8).init(allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "ops");
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "a", "Reshape", &.{"x"}, &.{"a_out"}));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "b", "Identity", &.{"a_out"}, &.{"b_out"}));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "c", "Reshape", &.{"b_out"}, &.{"y"}));

    var model = std.ArrayList(u8).init(allocator);
    errdefer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    return try model.toOwnedSlice();
}

fn nodeMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    op_type: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (inputs) |input| try appendStringField(&out, 1, input);
    for (outputs) |output| try appendStringField(&out, 2, output);
    try appendStringField(&out, 3, name);
    try appendStringField(&out, 4, op_type);
    return try out.toOwnedSlice();
}

fn appendOwnedMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []u8) !void {
    defer bytes.allocator.free(payload);
    try appendMessageField(bytes, field_number, payload);
}

fn appendMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, payload.len);
    try bytes.appendSlice(payload);
}

fn appendStringField(bytes: *std.ArrayList(u8), field_number: u64, value: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, value.len);
    try bytes.appendSlice(value);
}

fn appendVarintField(bytes: *std.ArrayList(u8), field_number: u64, value: u64) !void {
    try writeKey(bytes, field_number, 0);
    try writeVarint(bytes, value);
}

fn writeKey(bytes: *std.ArrayList(u8), field_number: u64, wire_type: u3) !void {
    try writeVarint(bytes, (field_number << 3) | wire_type);
}

fn writeVarint(bytes: *std.ArrayList(u8), raw: u64) !void {
    var value = raw;
    while (value >= 0x80) {
        try bytes.append(@intCast((value & 0x7f) | 0x80));
        value >>= 7;
    }
    try bytes.append(@intCast(value));
}
