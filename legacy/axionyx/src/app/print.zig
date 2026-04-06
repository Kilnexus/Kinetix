const std = @import("std");
const graph = @import("graph");
const runtime = @import("runtime");

pub fn printModelSummary(
    writer: anytype,
    graph_path: []const u8,
    weights_path: []const u8,
    model_graph: *const graph.Graph,
    support: *const runtime.SupportReport,
    first_tensor_data: []const f32,
) !void {
    const first_tensor = &model_graph.tensors[0];
    try writer.print("graph: {s}\n", .{graph_path});
    try writer.print("weights: {s}\n", .{weights_path});
    try writer.print("format_version: {d}\n", .{model_graph.format_version});
    try writer.print("model_name: {s}\n", .{model_graph.model_name});
    try writer.print("execution_nodes: {d}\n", .{model_graph.execution_nodes.len});
    try writer.print("tensor_count: {d}\n", .{model_graph.tensors.len});
    try writer.print("class_count: {d}\n", .{model_graph.class_count});
    try writer.print("detect_nodes: {d}\n", .{support.detect_nodes});
    try writer.print("runtime_compatible: {s}\n", .{if (support.supportsEndToEnd()) "true" else "false"});
    try writer.print(
        "supported_execution_nodes: {d}/{d}\n",
        .{ support.supported_execution_nodes, support.execution_nodes },
    );
    try writer.print(
        "supported_module_nodes: {d}/{d}\n",
        .{ support.supported_module_nodes, support.module_nodes },
    );
    try writer.print(
        "first_tensor: {s} len={d} first_value={d:.6}\n",
        .{ first_tensor.name, first_tensor_data.len, first_tensor_data[0] },
    );
    if (support.unsupported_execution_kinds.len > 0) {
        try writer.writeAll("unsupported_execution_kinds:");
        for (support.unsupported_execution_kinds) |entry| {
            try writer.print(" {s}({d})", .{ entry.kind, entry.count });
        }
        try writer.writeByte('\n');
    }
    if (support.unsupported_module_kinds.len > 0) {
        try writer.writeAll("unsupported_module_kinds:");
        for (support.unsupported_module_kinds) |entry| {
            try writer.print(" {s}({d})", .{ entry.kind, entry.count });
        }
        try writer.writeByte('\n');
    }
}

pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

pub fn bytesToMiB(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

pub fn detectBranchKindName(kind: runtime.DetectBranchKind) []const u8 {
    return switch (kind) {
        .generic => "generic",
        .cv2 => "cv2",
        .cv3 => "cv3",
    };
}

pub fn printMemoryStats(writer: anytype, stats: runtime.AllocationStats) !void {
    try writer.print(
        "memory: allocs={d} frees={d} resize={d} remap={d} failed={d} total_alloc_mb={d:.3} total_free_mb={d:.3} peak_live_mb={d:.3} live_end_mb={d:.3} outstanding={d}\n",
        .{
            stats.alloc_count,
            stats.free_count,
            stats.resize_count,
            stats.remap_count,
            stats.failed_alloc_count,
            bytesToMiB(stats.total_allocated_bytes),
            bytesToMiB(stats.total_freed_bytes),
            bytesToMiB(stats.peak_live_bytes),
            bytesToMiB(stats.live_bytes),
            stats.outstanding_allocations,
        },
    );
}
