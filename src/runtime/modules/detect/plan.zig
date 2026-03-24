const std = @import("std");
const graph = @import("graph");
const spec = @import("../../base/spec.zig");
const utils = @import("../../base/utils.zig");
const weights_mod = @import("weights");
const detect_types = @import("types.zig");

const BranchPlan = detect_types.BranchPlan;
const ConvPlan = detect_types.ConvPlan;

pub fn buildDetectBranchPlans(
    plans: []BranchPlan,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    branch: *const graph.ModuleNode,
) !void {
    if (plans.len != branch.children.len) return error.InvalidAttributeType;
    for (branch.children, 0..) |*node, index| {
        plans[index] = if (matchesDetectCv2Branch(node))
            .{ .cv2 = .{
                .conv0 = try buildConvPlan(model_graph, weights_blob, node.children[0].path),
                .conv0_fast = cv2PlanSupportsFast(&node.children[0]),
                .conv1 = try buildConvPlan(model_graph, weights_blob, node.children[1].path),
                .conv1_fast = cv2PlanSupportsFast(&node.children[1]),
                .head = try buildConvPlan(model_graph, weights_blob, node.children[2].path),
            } }
        else if (matchesDetectCv3Branch(node))
            .{ .cv3 = .{
                .stage0 = .{
                    .depthwise = try buildConvPlan(model_graph, weights_blob, node.children[0].children[0].path),
                    .depthwise_fast = cv3DepthwisePlanSupportsFast(&node.children[0].children[0]),
                    .pointwise = try buildConvPlan(model_graph, weights_blob, node.children[0].children[1].path),
                },
                .stage1 = .{
                    .depthwise = try buildConvPlan(model_graph, weights_blob, node.children[1].children[0].path),
                    .depthwise_fast = cv3DepthwisePlanSupportsFast(&node.children[1].children[0]),
                    .pointwise = try buildConvPlan(model_graph, weights_blob, node.children[1].children[1].path),
                },
                .head = try buildConvPlan(model_graph, weights_blob, node.children[2].path),
            } }
        else
            .{ .generic = node };
    }
}

pub fn buildConvPlan(
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
) !ConvPlan {
    const conv_spec = try spec.resolveConvSpec(model_graph, module_path);
    return convPlanFromSpec(weights_blob, conv_spec);
}

fn convPlanFromSpec(
    weights_blob: *const weights_mod.WeightsBlob,
    conv_spec: spec.ConvSpec,
) ConvPlan {
    return .{
        .weight = utils.tensorView(conv_spec.weight, weights_blob.slice(conv_spec.weight)),
        .bias = if (conv_spec.bias) |bias_meta| weights_blob.slice(bias_meta) else null,
        .stride_h = conv_spec.stride[0],
        .stride_w = conv_spec.stride[1],
        .pad_h = conv_spec.padding[0],
        .pad_w = conv_spec.padding[1],
        .groups = conv_spec.groups,
        .activation = conv_spec.activation,
    };
}

fn cv2PlanSupportsFast(module: *const graph.ModuleNode) bool {
    const cached = module.cached_conv;
    return cached.valid and
        cached.weight != null and
        cached.weight.?.shape[0] == 64 and
        cached.weight.?.shape[2] == 3 and
        cached.weight.?.shape[3] == 3 and
        cached.stride[0] == 1 and
        cached.stride[1] == 1 and
        cached.padding[0] == 1 and
        cached.padding[1] == 1 and
        cached.groups == 1 and
        cached.apply_silu;
}

fn cv3DepthwisePlanSupportsFast(module: *const graph.ModuleNode) bool {
    const cached = module.cached_conv;
    return cached.valid and
        cached.weight != null and
        cached.weight.?.shape[2] == 3 and
        cached.weight.?.shape[3] == 3 and
        cached.stride[0] == 1 and
        cached.stride[1] == 1 and
        cached.padding[0] == 1 and
        cached.padding[1] == 1 and
        cached.apply_silu and
        cached.groups == cached.weight.?.shape[0] and
        cached.weight.?.shape[1] == 1;
}

pub fn resolveDetectBranch(
    model_graph: *const graph.Graph,
    module_path: []const u8,
    primary: []const u8,
    fallback: []const u8,
) ?*const graph.ModuleNode {
    var branch_path_buffer: [256]u8 = undefined;
    const primary_path = utils.childModulePath(&branch_path_buffer, module_path, primary) catch return null;
    if (model_graph.findModule(primary_path)) |branch| return branch;

    var fallback_path_buffer: [256]u8 = undefined;
    const fallback_path = utils.childModulePath(&fallback_path_buffer, module_path, fallback) catch return null;
    if (model_graph.findModule(fallback_path)) |branch| return branch;

    return null;
}

pub fn resolveDetectBranchNode(
    module: *const graph.ModuleNode,
    primary: []const u8,
    fallback: []const u8,
) ?*const graph.ModuleNode {
    for (module.children) |*child| {
        if (isDirectChildNamed(module.path, child.path, primary)) return child;
    }
    for (module.children) |*child| {
        if (isDirectChildNamed(module.path, child.path, fallback)) return child;
    }
    return null;
}

pub fn matchesDetectCv2Branch(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 3 and
        std.mem.eql(u8, node.children[0].kind, "Conv") and
        std.mem.eql(u8, node.children[1].kind, "Conv") and
        std.mem.eql(u8, node.children[2].kind, "Conv2d");
}

pub fn matchesDetectCv3Stage(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 2 and
        std.mem.eql(u8, node.children[0].kind, "DWConv") and
        std.mem.eql(u8, node.children[1].kind, "Conv");
}

pub fn matchesDetectCv3Branch(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 3 and
        matchesDetectCv3Stage(&node.children[0]) and
        matchesDetectCv3Stage(&node.children[1]) and
        std.mem.eql(u8, node.children[2].kind, "Conv2d");
}

fn isDirectChildNamed(
    parent_path: []const u8,
    child_path: []const u8,
    name: []const u8,
) bool {
    if (child_path.len != parent_path.len + 1 + name.len) return false;
    if (!std.mem.startsWith(u8, child_path, parent_path)) return false;
    if (child_path[parent_path.len] != '.') return false;
    return std.mem.eql(u8, child_path[parent_path.len + 1 ..], name);
}
