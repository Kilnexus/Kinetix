const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const detect = @import("engine_vision_modules").detect;
const blocks = @import("engine_vision_modules").blocks;
const reuse_allocator = @import("engine_vision_reuse_allocator");
const types = @import("engine_vision_base").types;
const weights_mod = @import("weights");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;
pub const DetectOptions = detect.DetectOptions;
pub const DetectOutput = detect.DetectOutput;
pub const tensor_probe_count = 8;
pub const NodeProfile = struct {
    path: []const u8,
    kind: []const u8,
    elapsed_ns: u64,
    output_stats: ?TensorStats = null,
    detect_profile: ?detect.DetectProfile = null,
    c3k2_profile: ?blocks.C3k2Profile = null,
    sppf_profile: ?blocks.SPPFProfile = null,
};

pub const TensorStats = struct {
    shape: [4]usize,
    min: f32,
    max: f32,
    mean: f32,
    abs_max: f32,
    sum_abs: f64,
    probe_indices: [tensor_probe_count]usize,
    probe_values: [tensor_probe_count]f32,
};

pub const GraphProfile = struct {
    allocator: std.mem.Allocator,
    nodes: []NodeProfile,

    pub fn deinit(self: *GraphProfile) void {
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
};

const node_input_stack_limit = 8;

fn computeTensorStats(tensor: *const Tensor) TensorStats {
    const first = tensor.data[0];
    var min_value = first;
    var max_value = first;
    var sum: f64 = 0.0;
    var abs_max: f32 = @abs(first);
    var sum_abs: f64 = 0.0;

    for (tensor.data) |value| {
        min_value = @min(min_value, value);
        max_value = @max(max_value, value);
        abs_max = @max(abs_max, @abs(value));
        sum += value;
        sum_abs += @abs(@as(f64, value));
    }

    var probe_indices = std.mem.zeroes([tensor_probe_count]usize);
    var probe_values = std.mem.zeroes([tensor_probe_count]f32);
    const last_index = tensor.data.len - 1;
    for (0..tensor_probe_count) |probe_index| {
        const flat_index = if (tensor_probe_count == 1 or tensor.data.len == 1)
            0
        else
            @divTrunc(probe_index * last_index, tensor_probe_count - 1);
        probe_indices[probe_index] = flat_index;
        probe_values[probe_index] = tensor.data[flat_index];
    }

    return .{
        .shape = tensor.shape,
        .min = min_value,
        .max = max_value,
        .mean = @floatCast(sum / @as(f64, @floatFromInt(tensor.data.len))),
        .abs_max = abs_max,
        .sum_abs = sum_abs,
        .probe_indices = probe_indices,
        .probe_values = probe_values,
    };
}

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
            var feature_ptr_stack: [node_input_stack_limit]*const Tensor = undefined;
            const feature_ptrs = if (node.from.len <= feature_ptr_stack.len)
                feature_ptr_stack[0..node.from.len]
            else
                try scratch.alloc(*const Tensor, node.from.len);
            for (node.from, 0..) |source, source_index| {
                feature_ptrs[source_index] = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
            }
            const module = model_graph.execution_modules[node_index] orelse return error.ModuleNotFound;
            detect_output = try detect.runDetectNode(
                allocator,
                tensor_allocator,
                scratch,
                model_graph,
                weights_blob,
                module,
                feature_ptrs,
                detect_options,
            );
            releaseInputs(node.from, node_index, use_counts, outputs);
            continue;
        }

        if (std.mem.eql(u8, node.kind, "Concat")) {
            var tensor_ptr_stack: [node_input_stack_limit]*const Tensor = undefined;
            const tensor_ptrs = if (node.from.len <= tensor_ptr_stack.len)
                tensor_ptr_stack[0..node.from.len]
            else
                try scratch.alloc(*const Tensor, node.from.len);

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
            var feature_ptr_stack: [node_input_stack_limit]*const Tensor = undefined;
            const feature_ptrs = if (node.from.len <= feature_ptr_stack.len)
                feature_ptr_stack[0..node.from.len]
            else
                try scratch.alloc(*const Tensor, node.from.len);
            for (node.from, 0..) |source, source_index| {
                feature_ptrs[source_index] = resolveInput(source, node_index, input, outputs) orelse return error.ModuleNotFound;
            }
            const module = model_graph.execution_modules[node_index] orelse return error.ModuleNotFound;
            var profiled_detect = try detect.runDetectProfileNode(
                allocator,
                tensor_allocator,
                scratch,
                model_graph,
                weights_blob,
                module,
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
            var tensor_ptr_stack: [node_input_stack_limit]*const Tensor = undefined;
            const tensor_ptrs = if (node.from.len <= tensor_ptr_stack.len)
                tensor_ptr_stack[0..node.from.len]
            else
                try scratch.alloc(*const Tensor, node.from.len);

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
                .output_stats = computeTensorStats(&merged),
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
                    .output_stats = computeTensorStats(&output),
                };
            } else if (std.mem.eql(u8, node.kind, "C3k2")) {
                const profiled = try blocks.runC3k2ProfileNode(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = profiled.output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                    .output_stats = computeTensorStats(&profiled.output),
                    .c3k2_profile = profiled.c3k2_profile,
                };
            } else if (std.mem.eql(u8, node.kind, "SPPF")) {
                const profiled = try blocks.runSPPFProfileNode(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = profiled.output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                    .output_stats = computeTensorStats(&profiled.output),
                    .sppf_profile = profiled.sppf_profile,
                };
            } else {
                const output = try blocks.runModuleNodeDirect(tensor_allocator, model_graph, weights_blob, module, source);
                outputs[node_index] = output;
                profile_nodes[node_index] = .{
                    .path = node.path,
                    .kind = node.kind,
                    .elapsed_ns = timer.read(),
                    .output_stats = computeTensorStats(&output),
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
