const std = @import("std");
const detect_types = @import("types.zig");
const exec_common = @import("exec_common.zig");
const exec_fast = @import("exec_fast.zig");
const stopwatch = @import("engine_stopwatch");

const Tensor = detect_types.Tensor;
const Cv3StagePlan = detect_types.Cv3StagePlan;
const Cv3BranchPlan = detect_types.Cv3BranchPlan;
const DetectBranchProfile = detect_types.DetectBranchProfile;

fn runDetectCv3StagePlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv3StagePlan,
    input: *const Tensor,
) !Tensor {
    var hidden = if (plan.depthwise_fast)
        try exec_fast.runDetectFastDepthwise3x3Batch1(allocator, input, &plan.depthwise)
    else
        try exec_common.runConvPlan(allocator, &plan.depthwise, input);
    defer hidden.deinit();
    return try exec_common.runConvPlan(allocator, &plan.pointwise, &hidden);
}

pub fn runDetectCv3BranchPlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv3BranchPlan,
    input: *const Tensor,
) !Tensor {
    var hidden0 = try runDetectCv3StagePlanned(allocator, &plan.stage0, input);
    defer hidden0.deinit();
    var hidden1 = try runDetectCv3StagePlanned(allocator, &plan.stage1, &hidden0);
    defer hidden1.deinit();
    return try exec_common.runConvPlan(allocator, &plan.head, &hidden1);
}

pub fn runDetectCv3BranchPlannedProfile(
    allocator: std.mem.Allocator,
    plan: *const Cv3BranchPlan,
    input: *const Tensor,
    profile: *DetectBranchProfile,
) !Tensor {
    var timer = stopwatch.start();
    var hidden0 = if (plan.stage0.depthwise_fast)
        try exec_fast.runDetectFastDepthwise3x3Batch1(allocator, input, &plan.stage0.depthwise)
    else
        try exec_common.runConvPlan(allocator, &plan.stage0.depthwise, input);
    profile.stage0_ns = timer.read();
    profile.stage0_stats = exec_common.computeTensorStats(&hidden0);
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = try exec_common.runConvPlan(allocator, &plan.stage0.pointwise, &hidden0);
    profile.stage1_ns = timer.read();
    profile.stage1_stats = exec_common.computeTensorStats(&hidden1);
    defer hidden1.deinit();

    timer.reset();
    var hidden2 = if (plan.stage1.depthwise_fast)
        try exec_fast.runDetectFastDepthwise3x3Batch1(allocator, &hidden1, &plan.stage1.depthwise)
    else
        try exec_common.runConvPlan(allocator, &plan.stage1.depthwise, &hidden1);
    profile.stage2_ns = timer.read();
    profile.stage2_stats = exec_common.computeTensorStats(&hidden2);
    defer hidden2.deinit();

    timer.reset();
    var hidden3 = try exec_common.runConvPlan(allocator, &plan.stage1.pointwise, &hidden2);
    profile.stage3_ns = timer.read();
    profile.stage3_stats = exec_common.computeTensorStats(&hidden3);
    defer hidden3.deinit();

    timer.reset();
    const output = try exec_common.runConvPlan(allocator, &plan.head, &hidden3);
    profile.stage4_ns = timer.read();
    profile.stage4_stats = exec_common.computeTensorStats(&output);
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
