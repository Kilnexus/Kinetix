const std = @import("std");
const detect_types = @import("types.zig");
const exec_common = @import("exec_common.zig");
const exec_fast = @import("exec_fast.zig");

const Tensor = detect_types.Tensor;
const Cv2BranchPlan = detect_types.Cv2BranchPlan;
const DetectBranchProfile = detect_types.DetectBranchProfile;

pub fn runDetectCv2BranchPlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv2BranchPlan,
    input: *const Tensor,
) !Tensor {
    var hidden0 = if (plan.conv0_fast)
        try exec_fast.runDetectFast3x3Conv64Batch1(allocator, input, &plan.conv0)
    else
        try exec_common.runConvPlan(allocator, &plan.conv0, input);
    defer hidden0.deinit();
    var hidden1 = if (plan.conv1_fast)
        try exec_fast.runDetectFast3x3Conv64Batch1(allocator, &hidden0, &plan.conv1)
    else
        try exec_common.runConvPlan(allocator, &plan.conv1, &hidden0);
    defer hidden1.deinit();
    return try exec_common.runConvPlan(allocator, &plan.head, &hidden1);
}

pub fn runDetectCv2BranchPlannedProfile(
    allocator: std.mem.Allocator,
    plan: *const Cv2BranchPlan,
    input: *const Tensor,
    profile: *DetectBranchProfile,
) !Tensor {
    var timer = try std.time.Timer.start();
    var hidden0 = if (plan.conv0_fast)
        try exec_fast.runDetectFast3x3Conv64Batch1(allocator, input, &plan.conv0)
    else
        try exec_common.runConvPlan(allocator, &plan.conv0, input);
    profile.stage0_ns = timer.read();
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = if (plan.conv1_fast)
        try exec_fast.runDetectFast3x3Conv64Batch1(allocator, &hidden0, &plan.conv1)
    else
        try exec_common.runConvPlan(allocator, &plan.conv1, &hidden0);
    profile.stage1_ns = timer.read();
    defer hidden1.deinit();

    timer.reset();
    const output = try exec_common.runConvPlan(allocator, &plan.head, &hidden1);
    profile.stage2_ns = timer.read();
    return output;
}
