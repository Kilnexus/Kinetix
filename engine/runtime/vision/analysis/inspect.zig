const std = @import("std");
const graph = @import("graph");

pub const KindCount = struct {
    kind: []u8,
    count: usize,

    pub fn deinit(self: *KindCount, allocator: std.mem.Allocator) void {
        allocator.free(self.kind);
        self.* = undefined;
    }
};

pub const SupportReport = struct {
    allocator: std.mem.Allocator,
    model_name: []u8,
    class_count: i64,
    execution_nodes: usize,
    detect_nodes: usize,
    supported_execution_nodes: usize,
    module_nodes: usize,
    supported_module_nodes: usize,
    execution_kind_counts: []KindCount,
    module_kind_counts: []KindCount,
    unsupported_execution_kinds: []KindCount,
    unsupported_module_kinds: []KindCount,

    pub fn deinit(self: *SupportReport) void {
        self.allocator.free(self.model_name);
        deinitKindCounts(self.allocator, self.execution_kind_counts);
        deinitKindCounts(self.allocator, self.module_kind_counts);
        deinitKindCounts(self.allocator, self.unsupported_execution_kinds);
        deinitKindCounts(self.allocator, self.unsupported_module_kinds);
        self.* = undefined;
    }

    pub fn supportsEndToEnd(self: *const SupportReport) bool {
        return self.detect_nodes > 0 and
            self.supported_execution_nodes == self.execution_nodes and
            self.supported_module_nodes == self.module_nodes;
    }
};

pub fn inspectModel(allocator: std.mem.Allocator, model_graph: *const graph.Graph) !SupportReport {
    var execution_counts: std.ArrayListUnmanaged(KindCount) = .empty;
    errdefer deinitArrayListOwned(allocator, &execution_counts);
    var module_counts: std.ArrayListUnmanaged(KindCount) = .empty;
    errdefer deinitArrayListOwned(allocator, &module_counts);
    var unsupported_execution: std.ArrayListUnmanaged(KindCount) = .empty;
    errdefer deinitArrayListOwned(allocator, &unsupported_execution);
    var unsupported_module: std.ArrayListUnmanaged(KindCount) = .empty;
    errdefer deinitArrayListOwned(allocator, &unsupported_module);

    var detect_nodes: usize = 0;
    var supported_execution_nodes: usize = 0;
    for (model_graph.execution_nodes) |node| {
        try appendKindCount(allocator, &execution_counts, node.kind);
        if (std.mem.eql(u8, node.kind, "Detect")) detect_nodes += 1;
        if (isSupportedExecutionKind(node.kind)) {
            supported_execution_nodes += 1;
        } else {
            try appendKindCount(allocator, &unsupported_execution, node.kind);
        }
    }

    var module_nodes: usize = 0;
    var supported_module_nodes: usize = 0;
    try collectModuleKinds(
        allocator,
        &model_graph.module_tree,
        &module_counts,
        &unsupported_module,
        &module_nodes,
        &supported_module_nodes,
    );

    return .{
        .allocator = allocator,
        .model_name = try allocator.dupe(u8, model_graph.model_name),
        .class_count = model_graph.class_count,
        .execution_nodes = model_graph.execution_nodes.len,
        .detect_nodes = detect_nodes,
        .supported_execution_nodes = supported_execution_nodes,
        .module_nodes = module_nodes,
        .supported_module_nodes = supported_module_nodes,
        .execution_kind_counts = try execution_counts.toOwnedSlice(allocator),
        .module_kind_counts = try module_counts.toOwnedSlice(allocator),
        .unsupported_execution_kinds = try unsupported_execution.toOwnedSlice(allocator),
        .unsupported_module_kinds = try unsupported_module.toOwnedSlice(allocator),
    };
}

pub fn isSupportedExecutionKind(kind: []const u8) bool {
    return std.mem.eql(u8, kind, "Conv") or
        std.mem.eql(u8, kind, "DWConv") or
        std.mem.eql(u8, kind, "Conv2d") or
        std.mem.eql(u8, kind, "Bottleneck") or
        std.mem.eql(u8, kind, "SPPF") or
        std.mem.eql(u8, kind, "C3k") or
        std.mem.eql(u8, kind, "C3k2") or
        std.mem.eql(u8, kind, "Attention") or
        std.mem.eql(u8, kind, "PSABlock") or
        std.mem.eql(u8, kind, "C2PSA") or
        std.mem.eql(u8, kind, "Upsample") or
        std.mem.eql(u8, kind, "Concat") or
        std.mem.eql(u8, kind, "Detect");
}

pub fn isSupportedModuleKind(kind: []const u8) bool {
    return isSupportedExecutionKind(kind) or
        std.mem.eql(u8, kind, "DetectionModel") or
        std.mem.eql(u8, kind, "Sequential") or
        std.mem.eql(u8, kind, "ModuleList") or
        std.mem.eql(u8, kind, "MaxPool2d") or
        std.mem.eql(u8, kind, "SiLU") or
        std.mem.eql(u8, kind, "Identity") or
        std.mem.eql(u8, kind, "DFL");
}

fn collectModuleKinds(
    allocator: std.mem.Allocator,
    node: *const graph.ModuleNode,
    counts: *std.ArrayListUnmanaged(KindCount),
    unsupported: *std.ArrayListUnmanaged(KindCount),
    module_nodes: *usize,
    supported_module_nodes: *usize,
) !void {
    module_nodes.* += 1;
    try appendKindCount(allocator, counts, node.kind);
    if (isSupportedModuleKind(node.kind)) {
        supported_module_nodes.* += 1;
    } else {
        try appendKindCount(allocator, unsupported, node.kind);
    }

    for (node.children) |*child| {
        try collectModuleKinds(allocator, child, counts, unsupported, module_nodes, supported_module_nodes);
    }
}

fn appendKindCount(
    allocator: std.mem.Allocator,
    counts: *std.ArrayListUnmanaged(KindCount),
    kind: []const u8,
) !void {
    for (counts.items) |*entry| {
        if (std.mem.eql(u8, entry.kind, kind)) {
            entry.count += 1;
            return;
        }
    }
    try counts.append(allocator, .{
        .kind = try allocator.dupe(u8, kind),
        .count = 1,
    });
}

fn deinitArrayListOwned(allocator: std.mem.Allocator, counts: *std.ArrayListUnmanaged(KindCount)) void {
    deinitKindCounts(allocator, counts.items);
    counts.* = .empty;
}

fn deinitKindCounts(allocator: std.mem.Allocator, counts: []KindCount) void {
    for (counts) |*entry| entry.deinit(allocator);
    if (counts.len > 0) allocator.free(counts);
}
