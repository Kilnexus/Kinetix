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
    profile.stage0_stats = exec_common.computeTensorStats(&hidden0);
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = if (plan.conv1_fast)
        try exec_fast.runDetectFast3x3Conv64Batch1(allocator, &hidden0, &plan.conv1)
    else
        try exec_common.runConvPlan(allocator, &plan.conv1, &hidden0);
    profile.stage1_ns = timer.read();
    profile.stage1_stats = exec_common.computeTensorStats(&hidden1);
    defer hidden1.deinit();

    timer.reset();
    const output = try exec_common.runConvPlan(allocator, &plan.head, &hidden1);
    profile.stage2_ns = timer.read();
    profile.stage2_stats = exec_common.computeTensorStats(&output);
    profile.output_stats = exec_common.computeTensorStats(&output);
    profile.head_weight_stats = exec_common.computeTensorStats(&plan.head.weight);
    const top_weights = exec_common.topWeightClasses(&plan.head.weight);
    for (top_weights, 0..) |entry, index| {
        if (!std.math.isFinite(entry.value)) break;
        profile.head_top_weight_ids[index] = entry.class_id;
        profile.head_top_weight_values[index] = entry.value;
        profile.head_top_weight_count += 1;
    }
    if (plan.head.bias) |bias| {
        profile.head_bias_stats = exec_common.computeSliceStats(bias);
        const top_bias = exec_common.topBiasClasses(bias);
        for (top_bias, 0..) |entry, index| {
            if (!std.math.isFinite(entry.value)) break;
            profile.head_top_bias_ids[index] = entry.class_id;
            profile.head_top_bias_values[index] = entry.value;
            profile.head_top_bias_count += 1;
        }
    }
    return output;
}
