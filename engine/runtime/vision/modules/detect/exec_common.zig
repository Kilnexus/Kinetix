const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const blocks = @import("../blocks.zig");
const utils = @import("engine_vision_base").utils;
const weights_mod = @import("weights");
const detect_types = @import("types.zig");

const Tensor = detect_types.Tensor;
const ConvPlan = detect_types.ConvPlan;
pub const StatSummary = detect_types.StatSummary;
const max_branch_top_classes = detect_types.max_branch_top_classes;

pub fn runNodeChain(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    node: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor {
    if (std.mem.eql(u8, node.kind, "Sequential")) {
        if (node.children.len == 0) return input.clone();

        var current = try runNodeChain(allocator, model_graph, weights_blob, &node.children[0], input);
        for (node.children[1..]) |*child| {
            const next = try runNodeChain(allocator, model_graph, weights_blob, child, &current);
            current.deinit();
            current = next;
        }
        return current;
    }

    return blocks.runModule(allocator, model_graph, weights_blob, node.path, input);
}

pub fn runConvPlan(
    allocator: std.mem.Allocator,
    plan: *const ConvPlan,
    input: *const Tensor,
) !Tensor {
    const out_height = ((input.shape[2] + 2 * plan.pad_h - plan.weight.shape[2]) / plan.stride_h) + 1;
    const out_width = ((input.shape[3] + 2 * plan.pad_w - plan.weight.shape[3]) / plan.stride_w) + 1;

    var output = try Tensor.init(allocator, input.shape[0], plan.weight.shape[0], out_height, out_width);
    errdefer output.deinit();

    try ops.conv2d(input, &plan.weight, plan.bias, &output, .{
        .stride_h = plan.stride_h,
        .stride_w = plan.stride_w,
        .pad_h = plan.pad_h,
        .pad_w = plan.pad_w,
        .groups = plan.groups,
        .apply_silu = plan.activation == .silu,
    });
    if (plan.activation != .silu) {
        utils.applyActivation(&output, plan.activation);
    }
    return output;
}

pub fn computeSliceStats(values: []const f32) StatSummary {
    const first = values[0];
    var min_value = first;
    var max_value = first;
    var sum: f64 = 0.0;
    var abs_max: f32 = @abs(first);

    for (values) |value| {
        min_value = @min(min_value, value);
        max_value = @max(max_value, value);
        abs_max = @max(abs_max, @abs(value));
        sum += value;
    }

    return .{
        .min = min_value,
        .max = max_value,
        .mean = @floatCast(sum / @as(f64, @floatFromInt(values.len))),
        .abs_max = abs_max,
    };
}

pub fn computeTensorStats(tensor: *const Tensor) StatSummary {
    return computeSliceStats(tensor.data);
}

pub const RankedClassValue = struct {
    class_id: usize = 0,
    value: f32 = -std.math.inf(f32),
};

pub fn topBiasClasses(
    bias: []const f32,
) [max_branch_top_classes]RankedClassValue {
    var best = std.mem.zeroes([max_branch_top_classes]RankedClassValue);
    for (&best) |*entry| entry.* = .{};
    for (bias, 0..) |value, class_id| {
        updateRankedClasses(&best, class_id, value);
    }
    return best;
}

pub fn topWeightClasses(
    weight: *const Tensor,
) [max_branch_top_classes]RankedClassValue {
    var best = std.mem.zeroes([max_branch_top_classes]RankedClassValue);
    for (&best) |*entry| entry.* = .{};

    if (weight.shape[0] == 0) return best;
    const row_len = weight.data.len / weight.shape[0];
    for (0..weight.shape[0]) |class_id| {
        const start = class_id * row_len;
        const row = weight.data[start .. start + row_len];
        var abs_max: f32 = 0.0;
        for (row) |value| abs_max = @max(abs_max, @abs(value));
        updateRankedClasses(&best, class_id, abs_max);
    }
    return best;
}

fn updateRankedClasses(
    best: *[max_branch_top_classes]RankedClassValue,
    class_id: usize,
    value: f32,
) void {
    if (value <= best[best.len - 1].value) return;
    best[best.len - 1] = .{ .class_id = class_id, .value = value };
    var index = best.len - 1;
    while (index > 0 and best[index].value > best[index - 1].value) : (index -= 1) {
        const tmp = best[index - 1];
        best[index - 1] = best[index];
        best[index] = tmp;
    }
}
