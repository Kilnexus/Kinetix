const std = @import("std");
const graph = @import("graph");
const vision_base = @import("engine_vision_base");
const spec = vision_base.spec;
const utils = vision_base.utils;
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
const CachedDetectPlan = detect_types.CachedDetectPlan;
const max_detect_branch_levels = detect_types.max_detect_branch_levels;

const DetectCacheKey = struct {
    module_ptr: usize,
    weights_ptr: usize,
};

var detect_plan_cache_mutex: std.Thread.Mutex = .{};
var detect_plan_cache: std.AutoHashMapUnmanaged(DetectCacheKey, CachedDetectPlan) = .empty;

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

    const cached = try getCachedDetectPlan(model_graph, weights_blob, module);
    const nl = cached.nl;
    const nc = cached.nc;
    const reg_max = cached.reg_max;

    if (feature_inputs.len != nl or model_graph.strides.len != nl) return error.InvalidAttributeType;
    if (nl > max_detect_branch_levels) return error.InvalidAttributeType;
    const score_logit_threshold = postprocess.sigmoidThresholdToLogit(options.score_threshold);
    var profile = DetectProfile{};
    profile.level_count = nl;
    const reg_plans = cached.reg_plans[0..nl];
    const cls_plans = cached.cls_plans[0..nl];
    const dfl_weights = cached.dfl_weights;

    const max_candidates = maxCandidateCount(feature_inputs);
    const candidates = try scratch_allocator.alloc(Detection, max_candidates);
    defer scratch_allocator.free(candidates);
    var candidate_count: usize = 0;

    for (feature_inputs, 0..) |feature, level| {
        const level_candidate_start = candidate_count;
        profile.levels[level].max_class_logit = -std.math.inf(f32);
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
                    const best = bestClassForSpatial(cls.data, cls_batch_base, cls_plane, nc, spatial_index);
                    const best_logit = best.logit;
                    if (best_logit > profile.levels[level].max_class_logit) {
                        profile.levels[level].max_class_logit = best_logit;
                    }
                    if (best_logit < score_logit_threshold) continue;
                    const best_score = postprocess.sigmoid(best_logit);
                    if (best_score > profile.levels[level].max_class_score) {
                        profile.levels[level].max_class_score = best_score;
                    }

                    const anchor_x = (@as(f32, @floatFromInt(x)) + 0.5) * stride;
                    const anchor_y = (@as(f32, @floatFromInt(y)) + 0.5) * stride;

                    const left = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 0);
                    const top = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 1);
                    const right = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 2);
                    const bottom = postprocess.dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 3);

                    candidates[candidate_count] = .{
                        .x1 = anchor_x - left * stride,
                        .y1 = anchor_y - top * stride,
                        .x2 = anchor_x + right * stride,
                        .y2 = anchor_y + bottom * stride,
                        .score = best_score,
                        .class_id = best.class_id,
                    };
                    candidate_count += 1;
                }
            }
        }
        profile.levels[level].decode_ns = decode_timer.read();
        profile.levels[level].candidate_count = candidate_count - level_candidate_start;
        profile.decode_ns += profile.levels[level].decode_ns;
    }

    profile.candidate_count = candidate_count;
    var nms_timer = try std.time.Timer.start();
    const selected = try postprocess.nms(scratch_allocator, output_allocator, candidates[0..candidate_count], options);
    profile.nms_ns = nms_timer.read();
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

fn getCachedDetectPlan(
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
) !CachedDetectPlan {
    const key = DetectCacheKey{
        .module_ptr = @intFromPtr(module),
        .weights_ptr = @intFromPtr(weights_blob.data.ptr),
    };

    detect_plan_cache_mutex.lock();
    defer detect_plan_cache_mutex.unlock();

    if (detect_plan_cache.get(key)) |cached| return cached;

    const cached = try buildCachedDetectPlan(model_graph, weights_blob, module);
    try detect_plan_cache.put(std.heap.page_allocator, key, cached);
    return cached;
}

fn buildCachedDetectPlan(
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
) !CachedDetectPlan {
    const nl = module.cached_attrs.nl orelse @as(usize, @intCast(
        (module.getAttr("nl") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));
    const nc = module.cached_attrs.nc orelse @as(usize, @intCast(
        (module.getAttr("nc") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));
    const reg_max = module.cached_attrs.reg_max orelse @as(usize, @intCast(
        (module.getAttr("reg_max") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));
    if (nl > max_detect_branch_levels) return error.InvalidAttributeType;

    const reg_branch = plan.resolveDetectBranchNode(module, "cv2", "one2one_cv2") orelse return error.ModuleNotFound;
    const cls_branch = plan.resolveDetectBranchNode(module, "cv3", "one2one_cv3") orelse return error.ModuleNotFound;
    var cached = CachedDetectPlan{
        .nl = nl,
        .nc = nc,
        .reg_max = reg_max,
        .dfl_weights = null,
        .reg_plans = undefined,
        .cls_plans = undefined,
    };
    try plan.buildDetectBranchPlans(cached.reg_plans[0..nl], model_graph, weights_blob, reg_branch);
    try plan.buildDetectBranchPlans(cached.cls_plans[0..nl], model_graph, weights_blob, cls_branch);
    if (reg_max > 1) {
        const dfl_module = plan.resolveDetectBranchNode(module, "dfl", "dfl") orelse return error.ModuleNotFound;
        const dfl_conv = plan.resolveDetectBranchNode(dfl_module, "conv", "conv") orelse return error.ModuleNotFound;
        const dfl_spec = try spec.resolveConvSpecNode(model_graph, dfl_conv);
        cached.dfl_weights = weights_blob.slice(dfl_spec.weight);
    }
    return cached;
}

fn maxCandidateCount(feature_inputs: []const *const Tensor) usize {
    var total: usize = 0;
    for (feature_inputs) |feature| {
        total += feature.shape[0] * feature.shape[2] * feature.shape[3];
    }
    return total;
}

const BestClass = struct {
    logit: f32,
    class_id: usize,
};

fn bestClassForSpatial(
    cls_data: []const f32,
    cls_batch_base: usize,
    cls_plane: usize,
    nc: usize,
    spatial_index: usize,
) BestClass {
    var best_logit: f32 = -std.math.inf(f32);
    var best_class: usize = 0;
    var class_idx: usize = 0;

    while (class_idx + 3 < nc) : (class_idx += 4) {
        const logit0 = cls_data[cls_batch_base + class_idx * cls_plane + spatial_index];
        if (logit0 > best_logit) {
            best_logit = logit0;
            best_class = class_idx;
        }
        const logit1 = cls_data[cls_batch_base + (class_idx + 1) * cls_plane + spatial_index];
        if (logit1 > best_logit) {
            best_logit = logit1;
            best_class = class_idx + 1;
        }
        const logit2 = cls_data[cls_batch_base + (class_idx + 2) * cls_plane + spatial_index];
        if (logit2 > best_logit) {
            best_logit = logit2;
            best_class = class_idx + 2;
        }
        const logit3 = cls_data[cls_batch_base + (class_idx + 3) * cls_plane + spatial_index];
        if (logit3 > best_logit) {
            best_logit = logit3;
            best_class = class_idx + 3;
        }
    }

    while (class_idx < nc) : (class_idx += 1) {
        const logit = cls_data[cls_batch_base + class_idx * cls_plane + spatial_index];
        if (logit > best_logit) {
            best_logit = logit;
            best_class = class_idx;
        }
    }

    return .{ .logit = best_logit, .class_id = best_class };
}
