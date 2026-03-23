const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const spec = @import("../../base/spec.zig");
const types = @import("types.zig");
const utils = @import("../../base/utils.zig");

pub const Tensor = types.Tensor;

pub fn runConvModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runConvNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runConvNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    const conv_spec = try spec.resolveConvSpecNode(model_graph, module);
    const out_height = ((input.shape[2] + 2 * conv_spec.padding[0] - conv_spec.weight.shape[2]) / conv_spec.stride[0]) + 1;
    const out_width = ((input.shape[3] + 2 * conv_spec.padding[1] - conv_spec.weight.shape[3]) / conv_spec.stride[1]) + 1;

    var output = try Tensor.init(allocator, input.shape[0], conv_spec.weight.shape[0], out_height, out_width);
    errdefer output.deinit();

    var weight_tensor = utils.tensorView(conv_spec.weight, weights_blob.slice(conv_spec.weight));
    const bias_values = if (conv_spec.bias) |bias_meta| weights_blob.slice(bias_meta) else null;

    try ops.conv2d(input, &weight_tensor, bias_values, &output, .{
        .stride_h = conv_spec.stride[0],
        .stride_w = conv_spec.stride[1],
        .pad_h = conv_spec.padding[0],
        .pad_w = conv_spec.padding[1],
        .groups = conv_spec.groups,
        .apply_silu = conv_spec.activation == .silu,
    });
    if (conv_spec.activation != .silu) {
        utils.applyActivation(&output, conv_spec.activation);
    }
    return output;
}
