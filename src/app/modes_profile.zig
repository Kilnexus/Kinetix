const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");
const weights = @import("weights");
const app_print = @import("print.zig");
const vision = @import("../vision/preprocess.zig");
const vision_image = @import("../vision/image.zig");

pub fn runProfileMode(
    allocator: std.mem.Allocator,
    model_graph: *graph.Graph,
    weights_blob: *weights.WeightsBlob,
    image_path: []const u8,
    image_size: usize,
) !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();

    var src = try vision_image.loadRgb8(allocator, image_path);
    defer src.deinit();

    var prepared = try vision.prepareImageAsTensor(allocator, &src, image_size);
    defer prepared.deinit();

    var profile = try runtime.profileGraph(allocator, model_graph, weights_blob, &prepared.tensor, .{
        .score_threshold = 0.25,
        .iou_threshold = 0.7,
        .max_det = 300,
    });
    defer profile.deinit();

    var sorted = try allocator.dupe(runtime.NodeProfile, profile.nodes);
    defer allocator.free(sorted);
    std.mem.sort(runtime.NodeProfile, sorted, {}, struct {
        fn lessThan(_: void, lhs: runtime.NodeProfile, rhs: runtime.NodeProfile) bool {
            return lhs.elapsed_ns > rhs.elapsed_ns;
        }
    }.lessThan);

    var total_ns: u64 = 0;
    for (profile.nodes) |node| total_ns += node.elapsed_ns;

    try stdout.print("profile_image: {s}\n", .{image_path});
    try stdout.print("profile_size: {d}\n", .{image_size});
    try stdout.print("profile_total_ms: {d:.3}\n", .{app_print.nsToMs(total_ns)});
    try stdout.writeAll("profile_top_nodes_ms:\n");
    try stdout.writeAll("rank ms kind path\n");

    const top_n = @min(sorted.len, 8);
    for (sorted[0..top_n], 0..) |node, index| {
        try stdout.print(
            "{d} {d:.3} {s} {s}\n",
            .{ index + 1, app_print.nsToMs(node.elapsed_ns), node.kind, node.path },
        );
        if (node.detect_profile) |detect_profile| {
            try stdout.print(
                "detect_profile_ms: branch={d:.3} decode={d:.3} nms={d:.3} candidates={d} kept={d}\n",
                .{
                    app_print.nsToMs(detect_profile.branch_ns),
                    app_print.nsToMs(detect_profile.decode_ns),
                    app_print.nsToMs(detect_profile.nms_ns),
                    detect_profile.candidate_count,
                    detect_profile.kept_count,
                },
            );
            for (detect_profile.levels[0..detect_profile.level_count], 0..) |level_profile, level| {
                try stdout.print(
                    "detect_level_ms[{d}]: reg={d:.3} cls={d:.3} decode={d:.3}\n",
                    .{
                        level,
                        app_print.nsToMs(level_profile.reg_ns),
                        app_print.nsToMs(level_profile.cls_ns),
                        app_print.nsToMs(level_profile.decode_ns),
                    },
                );
                try stdout.print(
                    "detect_level_reg_detail[{d}]: kind={s} s0={d:.3} s1={d:.3} s2={d:.3} s3={d:.3} s4={d:.3}\n",
                    .{
                        level,
                        app_print.detectBranchKindName(level_profile.reg_detail.kind),
                        app_print.nsToMs(level_profile.reg_detail.stage0_ns),
                        app_print.nsToMs(level_profile.reg_detail.stage1_ns),
                        app_print.nsToMs(level_profile.reg_detail.stage2_ns),
                        app_print.nsToMs(level_profile.reg_detail.stage3_ns),
                        app_print.nsToMs(level_profile.reg_detail.stage4_ns),
                    },
                );
                try stdout.print(
                    "detect_level_cls_detail[{d}]: kind={s} s0={d:.3} s1={d:.3} s2={d:.3} s3={d:.3} s4={d:.3}\n",
                    .{
                        level,
                        app_print.detectBranchKindName(level_profile.cls_detail.kind),
                        app_print.nsToMs(level_profile.cls_detail.stage0_ns),
                        app_print.nsToMs(level_profile.cls_detail.stage1_ns),
                        app_print.nsToMs(level_profile.cls_detail.stage2_ns),
                        app_print.nsToMs(level_profile.cls_detail.stage3_ns),
                        app_print.nsToMs(level_profile.cls_detail.stage4_ns),
                    },
                );
            }
        }
        if (node.c3k2_profile) |c3k2_profile| {
            try stdout.print(
                "c3k2_profile_ms: cv1={d:.3} child={d:.3} concat={d:.3} cv2={d:.3} child_kind={s}\n",
                .{
                    app_print.nsToMs(c3k2_profile.cv1_ns),
                    app_print.nsToMs(c3k2_profile.child_ns),
                    app_print.nsToMs(c3k2_profile.concat_ns),
                    app_print.nsToMs(c3k2_profile.cv2_ns),
                    c3k2_profile.child_kind,
                },
            );
            if (c3k2_profile.child_c3k) |c3k_profile| {
                try stdout.print(
                    "c3k_profile_ms: cv1={d:.3} seq={d:.3} cv2={d:.3} concat={d:.3} cv3={d:.3} seq_kind={s}\n",
                    .{
                        app_print.nsToMs(c3k_profile.cv1_ns),
                        app_print.nsToMs(c3k_profile.seq_ns),
                        app_print.nsToMs(c3k_profile.cv2_ns),
                        app_print.nsToMs(c3k_profile.concat_ns),
                        app_print.nsToMs(c3k_profile.cv3_ns),
                        c3k_profile.seq_kind,
                    },
                );
            }
            if (c3k2_profile.child_bottleneck) |bottleneck_profile| {
                try stdout.print(
                    "bottleneck_profile_ms: cv1={d:.3} cv2={d:.3} add={d:.3} has_add={}\n",
                    .{
                        app_print.nsToMs(bottleneck_profile.cv1_ns),
                        app_print.nsToMs(bottleneck_profile.cv2_ns),
                        app_print.nsToMs(bottleneck_profile.add_ns),
                        bottleneck_profile.has_add,
                    },
                );
            }
        }
        if (node.sppf_profile) |sppf_profile| {
            try stdout.print(
                "sppf_profile_ms: cv1={d:.3} pool1={d:.3} pool2={d:.3} pool3={d:.3} concat={d:.3} cv2={d:.3}\n",
                .{
                    app_print.nsToMs(sppf_profile.cv1_ns),
                    app_print.nsToMs(sppf_profile.pool1_ns),
                    app_print.nsToMs(sppf_profile.pool2_ns),
                    app_print.nsToMs(sppf_profile.pool3_ns),
                    app_print.nsToMs(sppf_profile.concat_ns),
                    app_print.nsToMs(sppf_profile.cv2_ns),
                },
            );
        }
    }
}
