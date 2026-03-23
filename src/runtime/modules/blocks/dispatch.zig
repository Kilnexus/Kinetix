const std = @import("std");
const graph = @import("graph");
const weights_mod = @import("weights");
const bottleneck = @import("bottleneck.zig");
const c3k = @import("c3k.zig");
const c3k2 = @import("c3k2.zig");
const conv = @import("conv.zig");
const psa = @import("../psa.zig");
const sppf = @import("sppf.zig");
const types = @import("types.zig");

pub const Tensor = types.Tensor;

pub fn runModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) anyerror!Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runModuleNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runModuleNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor {
    if (std.mem.eql(u8, module.kind, "Identity")) {
        return input.clone();
    }
    if (std.mem.eql(u8, module.kind, "Sequential")) {
        if (module.children.len == 0) return input.clone();

        var current = try runModuleNode(allocator, model_graph, weights_blob, &module.children[0], input);
        errdefer current.deinit();
        for (module.children[1..]) |child| {
            const next = try runModuleNode(allocator, model_graph, weights_blob, &child, &current);
            current.deinit();
            current = next;
        }
        return current;
    }
    if (std.mem.eql(u8, module.kind, "Conv") or std.mem.eql(u8, module.kind, "DWConv") or std.mem.eql(u8, module.kind, "Conv2d")) {
        return conv.runConvNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "Bottleneck")) {
        return bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "SPPF")) {
        return sppf.runSPPFNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "C3k")) {
        return c3k.runC3kNode(allocator, model_graph, weights_blob, module, input, runModuleNode);
    }
    if (std.mem.eql(u8, module.kind, "C3k2")) {
        return c3k2.runC3k2Node(allocator, model_graph, weights_blob, module, input, runModuleNode);
    }
    if (std.mem.eql(u8, module.kind, "Attention")) {
        return psa.runAttention(allocator, model_graph, weights_blob, module.path, input);
    }
    if (std.mem.eql(u8, module.kind, "PSABlock")) {
        return psa.runPSABlock(allocator, model_graph, weights_blob, module.path, input);
    }
    if (std.mem.eql(u8, module.kind, "C2PSA")) {
        return psa.runC2PSA(allocator, model_graph, weights_blob, module.path, input);
    }
    return error.InvalidModuleKind;
}

pub fn runC3k2ProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !types.ProfiledTensor {
    return c3k2.runC3k2ProfileNode(allocator, model_graph, weights_blob, module, input, runModuleNode);
}
