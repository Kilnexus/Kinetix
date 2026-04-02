const std = @import("std");
const bert_mlm = @import("../../model/runtime/bert_mlm.zig");

pub fn fillMaskText(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    text: []const u8,
    top_k: usize,
) !void {
    var runtime = try bert_mlm.Runtime.init(allocator, model_dir);
    defer runtime.deinit();

    var result = try runtime.fillMask(text, top_k);
    defer result.deinit(allocator);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer fill-mask\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("text: {s}\n", .{text});
    try stdout.print("mask_position: {d}\n", .{result.mask_position});
    for (result.predictions, 0..) |prediction, idx| {
        try stdout.print(
            "prediction[{d}] token_id={d} token={s} logit={d:.6}\n",
            .{ idx, prediction.token_id, prediction.token, prediction.logit },
        );
    }
}
