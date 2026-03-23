const std = @import("std");
const runtime = @import("runtime");

pub fn writeDetectionsJson(path: []const u8, detections: *const runtime.DetectOutput) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;

    try out.writeAll("{\n");
    try out.print("  \"candidate_count\": {d},\n", .{detections.candidate_count});
    try out.writeAll("  \"detections\": [\n");
    for (detections.detections, 0..) |det, index| {
        try out.print(
            "    {{\"x1\": {d:.9}, \"y1\": {d:.9}, \"x2\": {d:.9}, \"y2\": {d:.9}, \"score\": {d:.9}, \"class_id\": {d}}}",
            .{ det.x1, det.y1, det.x2, det.y2, det.score, det.class_id },
        );
        if (index + 1 != detections.detections.len) {
            try out.writeAll(",\n");
        } else {
            try out.writeAll("\n");
        }
    }
    try out.writeAll("  ]\n}\n");
    try out.flush();
}

pub fn writeTraceJson(path: []const u8, trace: *const runtime.GraphTrace) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(&buffer);
    const out = &writer.interface;

    try out.writeAll("{\n  \"nodes\": [\n");
    for (trace.nodes, 0..) |node, index| {
        try out.print(
            "    {{\"index\": {d}, \"path\": \"{s}\", \"kind\": \"{s}\", \"shape\": [{d}, {d}, {d}, {d}], \"min\": {d:.9}, \"max\": {d:.9}, \"mean\": {d:.9}, \"l2\": {d:.9}, \"first\": {d:.9}}}",
            .{
                node.index,
                node.path,
                node.kind,
                node.shape[0],
                node.shape[1],
                node.shape[2],
                node.shape[3],
                node.min,
                node.max,
                node.mean,
                node.l2,
                node.first,
            },
        );
        if (index + 1 != trace.nodes.len) {
            try out.writeAll(",\n");
        } else {
            try out.writeAll("\n");
        }
    }
    try out.writeAll("  ]\n}\n");
    try out.flush();
}
