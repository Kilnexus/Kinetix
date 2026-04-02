const std = @import("std");
const bert_mlm = @import("../../model/runtime/bert_mlm.zig");

pub fn embedText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    text: []const u8,
    mode: bert_mlm.EmbeddingMode,
    count: usize,
) !void {
    var runtime = try bert_mlm.Runtime.init(allocator, model_dir);
    defer runtime.deinit();

    const embedding = try runtime.embedText(text, mode);
    defer allocator.free(embedding);

    const limit = @min(count, embedding.len);
    var norm: f32 = 0.0;
    for (embedding) |value| norm += value * value;
    norm = @sqrt(norm);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer embed-text\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("mode: {s}\n", .{mode.name()});
    try stdout.print("text: {s}\n", .{text});
    try stdout.print("dims: {d}\n", .{embedding.len});
    try stdout.print("l2_norm: {d:.6}\n", .{norm});
    try stdout.print("first_values: [", .{});
    for (embedding[0..limit], 0..) |value, idx| {
        if (idx != 0) try stdout.print(", ", .{});
        try stdout.print("{d:.6}", .{value});
    }
    try stdout.print("]\n", .{});
}
