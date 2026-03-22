const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const detect = @import("../modules/detect.zig");
const blocks = @import("../modules/blocks.zig");
const reuse_allocator = @import("../base/reuse_allocator.zig");
const types = @import("../base/types.zig");
const weights_mod = @import("weights");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;
pub const DetectOptions = detect.DetectOptions;
pub const DetectOutput = detect.DetectOutput;
pub const NodeProfile = struct {
    path: []const u8,
    kind: []const u8,
    elapsed_ns: u64,
    detect_profile: ?detect.DetectProfile = null,
    c3k2_profile: ?blocks.C3k2Profile = null,
    sppf_profile: ?blocks.SPPFProfile = null,
};

pub const GraphProfile = struct {
    allocator: std.mem.Allocator,
    nodes: []NodeProfile,

    pub fn deinit(self: *GraphProfile) void {
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
};

pub fn runUpsampleModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runUpsampleNode(allocator, model_graph, module, input);
}

fn runUpsampleNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    _ = model_graph;
    if (!std.mem.eql(u8, module.kind, "Upsample")) return error.InvalidModuleKind;

    const scale_value = (module.getAttr("scale_factor") orelse return error.MissingAttribute).asFloat() orelse return error.InvalidAttributeType;
    const scale: usize = @intFromFloat(scale_value);

    var output = try Tensor.init(
        allocator,
        input.shape[0],
        input.shape[1],
        input.shape[2] * scale,
        input.shape[3] * scale,
    );
    errdefer output.deinit();

    try ops.upsampleNearest(input, &output, scale, scale);
    return output;
}

pub fn runGraph(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    input: *const Tensor,
    detect_options: DetectOptions,
) !DetectOutput {
    var reuse = reuse_allocator.ReuseAllocator.init(allocator);
    defer reuse.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    return runGraphWithAllocators(
        allocator,
        reuse.allocator(),
        arena.allocator(),
        model_graph,
        weights_blob,
        input,
        detect_options,
    );
}

pub fn runGraphWithAllocators(
    allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    input: *const Tensor,
    detect_options: DetectOptions,
) !DetectOutput {
    var outputs = try scratch.alloc(?Tensor, model_graph.execution_nodes.len);
    for (outputs) |*item| item.* = null;
    const use_counts = try scratch.alloc(usize, model_graph.execution_use_counts.len);
    @memcpy(use_counts, model_graph.execution_use_counts);
    defer {
        for (outputs) |*item| {
            if (item.*) |*tensor| tensor.deinit();
        }
    }

    var detect_output: ?DetectOutput = null;

    for (model_graph.execution_nodes, 0..) |*node, node_index| {
        if (std.mem.eql(u8, node.kind, "Detect")) {
            var feature_ptrs = try scratch.alloc(*const Tensor, node.from.len);
            for (node.from, 0..) |source, source_index| {
                feature_ptrs[source_index] = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
            }
            var module_path_buffer: [256]u8 = undefined;
            detect_output = try detect.runDetect(
                allocator,
                tensor_allocator,
                scratch,
                model_graph,
                weights_blob,
                try modulePathForNode(&module_path_buffer, node.path),
                feature_ptrs,
                detect_options,
            );
            releaseInputs(node.from, node_index, use_counts, outputs);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "Concat")) {
            var tensor_ptrs = try scratch.alloc(*const Tensor, node.from.len);

            var channels: usize = 0;
            var height: usize = 0;
            var width: usize = 0;
            for (node.from, 0..) |source, source_index| {
                const tensor = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
                tensor_ptrs[source_index] = tensor;
                channels += tensor.shape[1];
                if (source_index == 0) {
                    height = tensor.shape[2];
                    width = tensor.shape[3];
                }
            }

            var merged = try Tensor.init(tensor_allocator, tensor_ptrs[0].shape[0], channels, height, width);
            errdefer merged.deinit();
            try ops.concatChannels(tensor_ptrs, &merged);
            outputs[node_index] = merged;
            releaseInputs(node.from, node_index, use_counts, outputs);
            continue;
        }

        const source = resolveInput(node.from[0], node_index, input, outputs) orelse return error.ModuleNotFound;
        const module = model_graph.execution_modules[node_index] orelse return error.ModuleNotFound;

        const output = if (std.mem.eql(u8, node.kind, "Upsample"))
            try runUpsampleNode(tensor_allocator, model_graph, module, source)
        else
            try blocks.runModuleNodeDirect(tensor_allocator, model_graph, weights_blob, module, source);
        outputs[node_index] = output;
        releaseInputs(node.from, node_index, use_counts, outputs);
    }

    return detect_output orelse error.ModuleNotFound;
}

pub fn profileGraph(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    input: *const Tensor,
    detect_options: DetectOptions,
) !GraphProfile {
    var reuse = reuse_allocator.ReuseAllocator.init(allocator);
    defer reuse.deinit();
    const tensor_allocator = reuse.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    var outputs = try scratch.alloc(?Tensor, model_graph.execution_nodes.len);
    for (outputs) |*item| item.* = null;
    const use_counts = try scratch.alloc(usize, model_graph.execution_use_counts.len);
    @memcpy(use_counts, model_graph.execution_use_counts);
    defer {
        for (outputs) |*item| {
            if (item.*) |*tensor| tensor.deinit();
        }
    }

    var profile_nodes = try allocator.alloc(NodeProfile, model_graph.execution_nodes.len);
    errdefer allocator.free(profile_nodes);

    for (model_graph.execution_nodes, 0..) |*node, node_index| {
        var timer = try std.time.Timer.start();

        if (std.mem.eql(u8, node.kind, "Detect")) {
            var feature_ptrs = try scratch.alloc(*const Tensor, node.from.len);
            for (node.from, 0..) |source, source_index| {
                feature_ptrs[source_index] = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
            }
            var module_path_buffer: [256]u8 = undefined;
            var profiled_detect = try detect.runDetectProfile(
                allocator,
                tensor_allocator,
                scratch,
                model_graph,
                weights_blob,
                try modulePathForNode(&module_path_buffer, node.path),
                feature_ptrs,
                detect_options,
            );
            profiled_detect.output.deinit();
            profile_nodes[node_index] = .{
                .path = node.path,
                .kind = node.kind,
                .elapsed_ns = timer.read(),
                .detect_profile = profiled_detect.profile,
            };
            releaseInputs(node.from, node_index, use_counts, outputs);
        } else if (std.mem.eql(u8, node.kind, "Concat")) {
            var tensor_ptrs = try scratch.alloc(*const Tensor, node.from.len);

            var channels: usize = 0;
            var height: usize = 0;
            var width: usize = 0;
            for (node.from, 0..) |source, source_index| {
                const tensor = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
                tensor_ptrs[source_index] = tensor;
                channels += tensor.shape[1];
                if (source_index == 0) {
                    height = tensor.shape[2];
                    width = tensor.shape[3];
                }
            }

            var merged = try Tensor.init(tensor_allocator, tensor_ptrs[0].shape[0], channels, height, width);
            errdefer merged.deinit();
            try ops.concatChannels(tensor_ptrs, &merged);
            outputs[node_index] = merged;
            profile_nodes[node_index] = .{
                .path = node.path,
                .kind = node.kind,
                .elapsed_ns = timer.read(),
            };
            releaseInputs(node.from, node_index, use_counts, outputs);
        } else {
            const source = resolveInput(node.from[0], node_index, input, outputs) orelse return error.ModuleNotFound;
            const module = model_graph.execution_modules[node_index] orelse return error.ModuleNotFound;

            if (std.mem.eql(u8, node.kind, "Upsample")) {
                const output = try runUpsampleNode(tensor_allocator, model_graph, module, source);
                outputs[node_index] = output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                };
            } else if (std.mem.eql(u8, node.kind, "C3k2")) {
                const profiled = try blocks.runC3k2ProfileNode(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = profiled.output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                    .c3k2_profile = profiled.c3k2_profile,
                };
            } else if (std.mem.eql(u8, node.kind, "SPPF")) {
                const profiled = try blocks.runSPPFProfileNode(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = profiled.output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                    .sppf_profile = profiled.sppf_profile,
                };
            } else {
                const output = try blocks.runModuleNodeDirect(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                };
            }
            releaseInputs(node.from, node_index, use_counts, outputs);
        }
    }

    return .{
        .allocator = allocator,
        .nodes = profile_nodes,
    };
}

pub fn resolveInput(
    from: i64,
    node_index: usize,
    input: *const Tensor,
    outputs: []?Tensor,
) ?*const Tensor {
    if (from == -1) {
        if (node_index == 0) return input;
        if (outputs[node_index - 1]) |*tensor| return tensor;
        return null;
    }

    const index: usize = @intCast(from);
    if (index >= outputs.len) return null;
    if (outputs[index]) |*tensor| return tensor;
    return null;
}

fn releaseInputs(
    from_list: []const i64,
    node_index: usize,
    use_counts: []usize,
    outputs: []?Tensor,
) void {
    for (from_list) |source| {
        const maybe_index: ?usize = if (source == -1)
            if (node_index > 0) node_index - 1 else null
        else
            @intCast(source);

        const index = maybe_index orelse continue;
        if (index >= outputs.len) continue;
        if (use_counts[index] == 0) continue;
        use_counts[index] -= 1;
        if (use_counts[index] == 0) {
            if (outputs[index]) |*tensor| {
                tensor.deinit();
                outputs[index] = null;
            }
        }
    }
}

pub fn modulePathForNode(buffer: []u8, path: []const u8) RuntimeError![]const u8 {
    if (!std.mem.startsWith(u8, path, "model.")) return error.InvalidAttributeType;
    return std.fmt.bufPrint(buffer, "model.model.{s}", .{path["model.".len..]}) catch return error.BufferTooSmall;
}
