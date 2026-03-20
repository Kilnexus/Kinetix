const std = @import("std");
const graph = @import("graph");
const weights_mod = @import("weights");
const detect_types = @import("types.zig");
const exec_common = @import("exec_common.zig");
const exec_cv2 = @import("exec_cv2.zig");
const exec_cv3 = @import("exec_cv3.zig");

const Tensor = detect_types.Tensor;
const BranchPlan = detect_types.BranchPlan;
const DetectBranchProfile = detect_types.DetectBranchProfile;

pub fn runDetectBranchPlan(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    plan: BranchPlan,
    input: *const Tensor,
) !Tensor {
    return switch (plan) {
        .cv2 => |cv2| exec_cv2.runDetectCv2BranchPlanned(allocator, &cv2, input),
        .cv3 => |cv3| exec_cv3.runDetectCv3BranchPlanned(allocator, &cv3, input),
        .generic => |node| exec_common.runNodeChain(allocator, model_graph, weights_blob, node, input),
    };
}

pub fn runDetectBranchPlanProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    plan: BranchPlan,
    input: *const Tensor,
    profile: *DetectBranchProfile,
) !Tensor {
    return switch (plan) {
        .cv2 => |cv2| blk: {
            profile.kind = .cv2;
            break :blk exec_cv2.runDetectCv2BranchPlannedProfile(allocator, &cv2, input, profile);
        },
        .cv3 => |cv3| blk: {
            profile.kind = .cv3;
            break :blk exec_cv3.runDetectCv3BranchPlannedProfile(allocator, &cv3, input, profile);
        },
        .generic => |node| blk: {
            profile.kind = .generic;
            break :blk exec_common.runNodeChain(allocator, model_graph, weights_blob, node, input);
        },
    };
}
