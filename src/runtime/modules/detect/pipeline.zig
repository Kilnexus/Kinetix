const std = @import("std");
const graph = @import("graph");
const spec = @import("../../base/spec.zig");
const utils = @import("../../base/utils.zig");
const weights_mod = @import("weights");
const detect_types = @import("types.zig");
const plan = @import("plan.zig");
const branch_exec = @import("branch_exec.zig");
const postprocess = @import("postprocess.zig");

const Tensor = detect_types.Tensor;
const Detection = detect_types.Detection;
const DetectOptions = detect_types.DetectOptions;
const DetectOutput = detect_types.DetectOutput;
const DetectProfile = detect_types.DetectProfile;
const DetectBranchProfile = detect_types.DetectBranchProfile;
const ProfiledDetectOutput = detect_types.ProfiledDetectOutput;
const BranchPlan = detect_types.BranchPlan;
const max_detect_branch_levels = detect_types.max_detect_branch_levels;

pub fn runDetect(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !DetectOutput {
    const profiled = try runDetectProfile(
        output_allocator,
        tensor_allocator,
        scratch_allocator,
        model_graph,
        weights_blob,
        module_path,
        feature_inputs,
        options,
    );
    return profiled.output;
}

pub fn runDetectNode(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !DetectOutput {
    const profiled = try runDetectProfileNode(
        output_allocator,
        tensor_allocator,
        scratch_allocator,
        model_graph,
        weights_blob,
        module,
        feature_inputs,
        options,
    );
    return profiled.output;
}

pub fn runDetectProfile(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !ProfiledDetectOutput {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "Detect")) return error.InvalidModuleKind;
    return runDetectProfileNode(
        output_allocator,
        tensor_allocator,
        scratch_allocator,
        model_graph,
        weights_blob,
        module,
        feature_inputs,
        options,
    );
}

pub fn runDetectProfileNode(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !ProfiledDetectOutput {
    if (!std.mem.eql(u8, module.kind, "Detect")) return error.InvalidModuleKind;

    const nl = module.cached_attrs.nl orelse @as(usize, @intCast(
        (module.getAttr("nl") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));
    const nc = module.cached_attrs.nc orelse @as(usize, @intCast(
        (module.getAttr("nc") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));
    const reg_max = module.cached_attrs.reg_max orelse @as(usize, @intCast(
        (module.getAttr("reg_max") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));

    if (feature_inputs.len != nl or model_graph.strides.len != nl) return error.InvalidAttributeType;
    if (nl > max_detect_branch_levels) return error.InvalidAttributeType;

    const reg_branch = plan.resolveDetectBranchNode(module, "cv2", "one2one_cv2") orelse return error.ModuleNotFound;
    const cls_branch = plan.resolveDetectBranchNode(module, "cv3", "one2one_cv3") orelse return error.ModuleNotFound;
    var reg_plan_storage: [max_detect_branch_levels]BranchPlan = undefined;
    var cls_plan_storage: [max_detect_branch_levels]BranchPlan = undefined;
    const reg_plans = reg_plan_storage[0..nl];
    const cls_plans = cls_plan_storage[0..nl];
    try plan.buildDetectBranchPlans(reg_plans, model_graph, weights_blob, reg_branch);
    try plan.buildDetectBranchPlans(cls_plans, model_graph, weights_blob, cls_branch);
    const dfl_weights = if (reg_max > 1) blk: {
        const dfl_module = plan.resolveDetectBranchNode(module, "dfl", "dfl") orelse return error.ModuleNotFound;
        const dfl_conv = plan.resolveDetectBranchNode(dfl_module, "conv", "conv") orelse return error.ModuleNotFound;
        const dfl_spec = try spec.resolveConvSpecNode(model_graph, dfl_conv);
        break :blk weights_blob.slice(dfl_spec.weight);
    } else null;
    const score_logit_threshold = postprocess.sigmoidThresholdToLogit(options.score_threshold);
    var profile = DetectProfile{};
    profile.level_count = nl;

    var candidates: std.ArrayListUnmanaged(Detection) = .empty;
    errdefer candidates.deinit(scratch_allocator);

    for (feature_inputs, 0..) |feature, level| {
        var reg_profile = DetectBranchProfile{};
        var timer = try std.time.Timer.start();
        var reg = try branch_exec.runDetectBranchPlanProfile(tensor_allocator, model_graph, weights_blob, reg_plans[level], feature, &reg_profile);
        profile.levels[level].reg_ns = timer.read();
        profile.levels[level].reg_detail = reg_profile;
        defer reg.deinit();
        var cls_profile = DetectBranchProfile{};
        timer.reset();
        var cls = try branch_exec.runDetectBranchPlanProfile(tensor_allocator, model_graph, weights_blob, cls_plans[level], feature, &cls_profile);
        profile.levels[level].cls_ns = timer.read();
        profile.levels[level].cls_detail = cls_profile;
        defer cls.deinit();
        profile.branch_ns += profile.levels[level].reg_ns + profile.levels[level].cls_ns;

        if (reg.shape[0] != cls.shape[0] or reg.shape[2] != cls.shape[2] or reg.shape[3] != cls.shape[3]) {
            return error.InvalidAttributeType;
        }

        var decode_timer = try std.time.Timer.start();
        const stride = model_graph.strides[level];
        const reg_plane = reg.shape[2] * reg.shape[3];
        const cls_plane = cls.shape[2] * cls.shape[3];
        for (0..reg.shape[0]) |n| {
            const reg_batch_base = n * reg.shape[1] * reg_plane;
            const cls_batch_base = n * cls.shape[1] * cls_plane;
            for (0..reg.shape[2]) |y| {
                for (0..reg.shape[3]) |x| {
                    const spatial_index = y * reg.shape[3] + x;
                    var best_logit: f32 = -std.math.inf(f32);
                    var best_class: usize = 0;
                    for (0..nc) |class_idx| {
                        const logit = cls.data[cls_batch_base + class_idx * cls_plane + spatial_index];
                        if (logit > best_logit) {
                            best_logit = logit;
                            best_class = class_idx;
                        }
                    }
                    if (best_logit < score_logit_threshold) continue;
                    const best_score = postprocess.sigmoid(best_logit);

                    const anchor_x = (@as(f32, @floatFromInt(x)) + 0.5) * stride;
                    const anchor_y = (@as(f32, @floatFromInt(y)) + 0.5) * stride;

                    const left = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 0);
                    const top = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 1);
                    const right = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 2);
                    const bottom = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 3);

                    try candidates.append(scratch_allocator, .{
                        .x1 = anchor_x - left * stride,
                        .y1 = anchor_y - top * stride,
                        .x2 = anchor_x + right * stride,
                        .y2 = anchor_y + bottom * stride,
                        .score = best_score,
                        .class_id = best_class,
                    });
                }
            }
        }
        profile.levels[level].decode_ns = decode_timer.read();
        profile.decode_ns += profile.levels[level].decode_ns;
    }

    const candidate_count = candidates.items.len;
    profile.candidate_count = candidate_count;
    var nms_timer = try std.time.Timer.start();
    const selected = try postprocess.nms(scratch_allocator, output_allocator, candidates.items, options);
    profile.nms_ns = nms_timer.read();
    candidates.deinit(scratch_allocator);
    profile.kept_count = selected.len;

    return .{
        .output = .{
            .allocator = output_allocator,
            .detections = selected,
            .candidate_count = candidate_count,
        },
        .profile = profile,
    };
}
