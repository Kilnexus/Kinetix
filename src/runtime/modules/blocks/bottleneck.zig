const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const conv = @import("conv.zig");
const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const BottleneckProfiledTensor = types.BottleneckProfiledTensor;

pub fn runBottleneck(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runBottleneckNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runBottleneckNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "Bottleneck")) return error.InvalidModuleKind;
    return runBottleneckNodeUnchecked(allocator, model_graph, weights_blob, module, input);
}

pub fn runBottleneckNodeUnchecked(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    var hidden = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer hidden.deinit();

    var output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &hidden);
    const has_add = module.cached_attrs.add orelse
        ((module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType);
    if (has_add) {
        ops.addInPlaceUnchecked(&output, input);
    }
    return output;
}

pub fn runBottleneckPairNodesUnchecked(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    first: *const graph.ModuleNode,
    second: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    var current = try runBottleneckNodeUnchecked(allocator, model_graph, weights_blob, first, input);
    errdefer current.deinit();
    const next = try runBottleneckNodeUnchecked(allocator, model_graph, weights_blob, second, &current);
    current.deinit();
    return next;
}

pub fn runBottleneckProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !BottleneckProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runBottleneckProfileNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runBottleneckProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !BottleneckProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "Bottleneck")) return error.InvalidModuleKind;
    return runBottleneckProfileNodeUnchecked(allocator, model_graph, weights_blob, module, input);
}

pub fn runBottleneckProfileNodeUnchecked(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !BottleneckProfiledTensor {
    var profile = types.BottleneckProfile{};
    var timer = try std.time.Timer.start();

    var hidden = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer hidden.deinit();

    timer.reset();
    var output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &hidden);
    profile.cv2_ns = timer.read();

    const has_add = module.cached_attrs.add orelse
        ((module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType);
    profile.has_add = has_add;
    if (has_add) {
        timer.reset();
        ops.addInPlaceUnchecked(&output, input);
        profile.add_ns = timer.read();
    }
    return .{ .output = output, .bottleneck_profile = profile };
}
