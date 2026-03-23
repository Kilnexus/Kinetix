const std = @import("std");
const graph = @import("graph");
const weights_mod = @import("weights");
const bottleneck = @import("blocks/bottleneck.zig");
const c3k = @import("blocks/c3k.zig");
const c3k2 = @import("blocks/c3k2.zig");
const conv = @import("blocks/conv.zig");
const dispatch = @import("blocks/dispatch.zig");
const sppf = @import("blocks/sppf.zig");
const types = @import("blocks/types.zig");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;
pub const C3k2Profile = types.C3k2Profile;
pub const C3kProfile = types.C3kProfile;
pub const BottleneckProfile = types.BottleneckProfile;
pub const SPPFProfile = types.SPPFProfile;
pub const ProfiledTensor = types.ProfiledTensor;
pub const BottleneckProfiledTensor = types.BottleneckProfiledTensor;
pub const SPPFProfiledTensor = types.SPPFProfiledTensor;
pub const C3kProfiledTensor = types.C3kProfiledTensor;

pub const runConvModule = conv.runConvModule;
pub const runBottleneck = bottleneck.runBottleneck;
pub const runBottleneckProfile = bottleneck.runBottleneckProfile;
pub const runSPPF = sppf.runSPPF;
pub const runSPPFNode = sppf.runSPPFNode;
pub const runSPPFProfile = sppf.runSPPFProfile;
pub const runSPPFProfileNode = sppf.runSPPFProfileNode;
pub const runModule = dispatch.runModule;
pub const runModuleNodeDirect = dispatch.runModuleNode;
pub const runC3k2ProfileNode = dispatch.runC3k2ProfileNode;

pub fn runC3k(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return c3k.runC3kNode(allocator, model_graph, weights_blob, module, input, dispatch.runModuleNode);
}

pub fn runC3kProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !C3kProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return c3k.runC3kProfileNode(allocator, model_graph, weights_blob, module, input, dispatch.runModuleNode);
}

pub fn runC3k2(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return c3k2.runC3k2Node(allocator, model_graph, weights_blob, module, input, dispatch.runModuleNode);
}

pub fn runC3k2Profile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !ProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return c3k2.runC3k2ProfileNode(allocator, model_graph, weights_blob, module, input, dispatch.runModuleNode);
}
