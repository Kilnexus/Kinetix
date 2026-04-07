const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const blocks = @import("../blocks.zig");
const utils = @import("engine_vision_base").utils;
const weights_mod = @import("weights");
const detect_types = @import("types.zig");

const Tensor = detect_types.Tensor;
const ConvPlan = detect_types.ConvPlan;

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
