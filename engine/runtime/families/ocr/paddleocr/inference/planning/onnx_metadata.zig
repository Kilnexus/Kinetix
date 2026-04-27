const std = @import("std");
const shared_graph = @import("shared_graph");
const shared_ops = @import("shared_ops");

const io = std.Options.debug_io;

pub const max_unsupported_op_entries: usize = 32;

pub const UnsupportedOpEntry = struct {
    op_type: []const u8 = "",
    count: usize = 0,
};

pub const Summary = struct {
    loaded_graph_count: usize = 0,
    det_graph_count: usize = 0,
    rec_graph_count: usize = 0,
    cls_graph_count: usize = 0,
    total_node_count: usize = 0,
    total_initializer_count: usize = 0,
    external_initializer_count: usize = 0,
    supported_node_count: usize = 0,
    unsupported_node_count: usize = 0,
    unsupported_ops: [max_unsupported_op_entries]UnsupportedOpEntry = [_]UnsupportedOpEntry{.{}} ** max_unsupported_op_entries,
    unsupported_op_entry_count: usize = 0,
};

pub fn inspect(allocator: std.mem.Allocator, model_dir: []const u8) !Summary {
    var summary = Summary{};
    try inspectDir(allocator, model_dir, "", &summary, 0);
    return summary;
}

fn inspectDir(
    allocator: std.mem.Allocator,
    root: []const u8,
    relative: []const u8,
    summary: *Summary,
    depth: usize,
) !void {
    if (depth > 4) return;

    const dir_path = if (relative.len == 0)
        try allocator.dupe(u8, root)
    else
        try std.fs.path.join(allocator, &.{ root, relative });
    defer allocator.free(dir_path);

    var dir = if (std.fs.path.isAbsolute(dir_path))
        std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return
    else
        std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        switch (entry.kind) {
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".onnx")) continue;
                const onnx_relative = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                defer allocator.free(onnx_relative);
                try addGraph(allocator, root, onnx_relative, summary);
            },
            .directory => {
                const child = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                defer allocator.free(child);
                try inspectDir(allocator, root, child, summary, depth + 1);
            },
            else => {},
        }
    }
}

fn addGraph(
    allocator: std.mem.Allocator,
    root: []const u8,
    relative_path: []const u8,
    summary: *Summary,
) !void {
    const full_path = try std.fs.path.join(allocator, &.{ root, relative_path });
    defer allocator.free(full_path);

    var metadata = shared_graph.onnx.metadata.loadFromFile(allocator, full_path) catch return;
    defer metadata.deinit();

    summary.loaded_graph_count += 1;
    summary.total_node_count += metadata.graph.nodes.len;
    addStageCount(summary, relative_path);
    addInitializers(summary, &metadata);
    addOperatorSupport(summary, &metadata, stageName(relative_path));
}

fn addStageCount(summary: *Summary, path: []const u8) void {
    if (std.ascii.indexOfIgnoreCase(path, "det") != null) {
        summary.det_graph_count += 1;
    } else if (std.ascii.indexOfIgnoreCase(path, "rec") != null) {
        summary.rec_graph_count += 1;
    } else if (std.ascii.indexOfIgnoreCase(path, "cls") != null) {
        summary.cls_graph_count += 1;
    }
}

fn stageName(path: []const u8) []const u8 {
    if (std.ascii.indexOfIgnoreCase(path, "det") != null) return "det";
    if (std.ascii.indexOfIgnoreCase(path, "rec") != null) return "rec";
    if (std.ascii.indexOfIgnoreCase(path, "cls") != null) return "cls";
    return "ocr";
}

fn addInitializers(summary: *Summary, metadata: *const shared_graph.onnx.metadata.ModelMetadata) void {
    summary.total_initializer_count += metadata.graph.initializers.len;
    for (metadata.graph.initializers) |initializer| {
        if (initializer.isExternal()) summary.external_initializer_count += 1;
    }
}

fn addOperatorSupport(summary: *Summary, metadata: *const shared_graph.onnx.metadata.ModelMetadata, stage: []const u8) void {
    for (metadata.graph.nodes) |node| {
        if (shared_ops.graph.isSupported(node.op_type)) {
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
    const interned = internUnsupportedOpKey(key);
    for (summary.unsupported_ops[0..summary.unsupported_op_entry_count]) |*entry| {
        if (std.mem.eql(u8, entry.op_type, interned)) {
            entry.count += 1;
            sortUnsupportedOps(summary);
            return;
        }
    }
    if (summary.unsupported_op_entry_count >= summary.unsupported_ops.len) return;
    const index = summary.unsupported_op_entry_count;
    summary.unsupported_ops[index] = .{ .op_type = interned, .count = 1 };
    summary.unsupported_op_entry_count += 1;
    sortUnsupportedOps(summary);
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
    "det:Resize",
    "det:Pad",
    "det:ReduceMean",
    "det:BatchNormalization",
    "det:HardSwish",
    "det:ArgMax",
    "det:Where",
    "det:Expand",
    "det:Split",
    "rec:Resize",
    "rec:Pad",
    "rec:ReduceMean",
    "rec:BatchNormalization",
    "rec:HardSwish",
    "rec:ArgMax",
    "rec:Where",
    "rec:Expand",
    "rec:Split",
    "cls:Resize",
    "cls:Pad",
    "cls:ReduceMean",
    "cls:BatchNormalization",
    "cls:HardSwish",
    "cls:ArgMax",
    "cls:Where",
    "cls:Expand",
    "cls:Split",
};

test "paddleocr onnx metadata planning tolerates missing graphs" {
    const summary = try inspect(std.testing.allocator, "missing-paddleocr-dir");
    try std.testing.expectEqual(@as(usize, 0), summary.loaded_graph_count);
}

test "paddleocr onnx metadata planning aggregates unsupported ops" {
    var summary = Summary{};
    var metadata = try shared_graph.onnx.metadata.parseModel(std.testing.allocator, try tinyUnsupportedModel(std.testing.allocator));
    defer metadata.deinit();
    addOperatorSupport(&summary, &metadata, "det");

    try std.testing.expectEqual(@as(usize, 1), summary.supported_node_count);
    try std.testing.expectEqual(@as(usize, 2), summary.unsupported_node_count);
    try std.testing.expectEqual(@as(usize, 1), summary.unsupported_op_entry_count);
    try std.testing.expectEqualStrings("det:Where", summary.unsupported_ops[0].op_type);
    try std.testing.expectEqual(@as(usize, 2), summary.unsupported_ops[0].count);
}

fn tinyUnsupportedModel(allocator: std.mem.Allocator) ![]u8 {
    var graph = std.ArrayList(u8).init(allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "ops");
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "a", "Where", &.{"x"}, &.{"a_out"}));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "b", "Identity", &.{"a_out"}, &.{"b_out"}));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "c", "Where", &.{"b_out"}, &.{"y"}));

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
    while (value >= 0x80) : (value >>= 7) {
        try bytes.append(@as(u8, @intCast(value & 0x7f)) | 0x80);
    }
    try bytes.append(@intCast(value));
}
