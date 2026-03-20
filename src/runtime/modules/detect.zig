const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const blocks = @import("blocks.zig");
const spec = @import("../base/spec.zig");
const types = @import("../base/types.zig");
const utils = @import("../base/utils.zig");
const weights_mod = @import("weights");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;

pub const Detection = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
};

pub const DetectOptions = struct {
    score_threshold: f32 = 0.25,
    iou_threshold: f32 = 0.7,
    max_det: usize = 300,
};

pub const DetectOutput = struct {
    allocator: std.mem.Allocator,
    detections: []Detection,
    candidate_count: usize,

    pub fn deinit(self: *DetectOutput) void {
        self.allocator.free(self.detections);
        self.* = undefined;
    }
};

pub const DetectProfile = struct {
    branch_ns: u64 = 0,
    decode_ns: u64 = 0,
    nms_ns: u64 = 0,
    candidate_count: usize = 0,
    kept_count: usize = 0,
    level_count: usize = 0,
    levels: [max_detect_branch_levels]DetectLevelProfile = std.mem.zeroes([max_detect_branch_levels]DetectLevelProfile),
};

pub const DetectLevelProfile = struct {
    reg_ns: u64 = 0,
    cls_ns: u64 = 0,
    decode_ns: u64 = 0,
    reg_detail: DetectBranchProfile = .{},
    cls_detail: DetectBranchProfile = .{},
};

pub const DetectBranchProfile = struct {
    kind: DetectBranchKind = .generic,
    stage0_ns: u64 = 0,
    stage1_ns: u64 = 0,
    stage2_ns: u64 = 0,
    stage3_ns: u64 = 0,
    stage4_ns: u64 = 0,
};

pub const DetectBranchKind = enum {
    generic,
    cv2,
    cv3,
};

pub const ProfiledDetectOutput = struct {
    output: DetectOutput,
    profile: DetectProfile,
};

const ConvPlan = struct {
    weight: Tensor,
    bias: ?[]const f32,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    groups: usize,
    activation: types.Activation,
};

const Cv2BranchPlan = struct {
    conv0: ConvPlan,
    conv1: ConvPlan,
    head: ConvPlan,
};

const Cv3StagePlan = struct {
    depthwise: ConvPlan,
    pointwise: ConvPlan,
};

const Cv3BranchPlan = struct {
    stage0: Cv3StagePlan,
    stage1: Cv3StagePlan,
    head: ConvPlan,
};

const BranchPlan = union(enum) {
    cv2: Cv2BranchPlan,
    cv3: Cv3BranchPlan,
    generic: *const graph.ModuleNode,
};

const max_detect_branch_levels = 8;
const max_detect_fast_threads = 2;

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

    const nl: usize = @intCast(
        (module.getAttr("nl") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );
    const nc: usize = @intCast(
        (module.getAttr("nc") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );
    const reg_max: usize = @intCast(
        (module.getAttr("reg_max") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );

    if (feature_inputs.len != nl or model_graph.strides.len != nl) return error.InvalidAttributeType;
    if (nl > max_detect_branch_levels) return error.InvalidAttributeType;

    const reg_branch = resolveDetectBranch(model_graph, module_path, "cv2", "one2one_cv2") orelse return error.ModuleNotFound;
    const cls_branch = resolveDetectBranch(model_graph, module_path, "cv3", "one2one_cv3") orelse return error.ModuleNotFound;
    var reg_plan_storage: [max_detect_branch_levels]BranchPlan = undefined;
    var cls_plan_storage: [max_detect_branch_levels]BranchPlan = undefined;
    const reg_plans = reg_plan_storage[0..nl];
    const cls_plans = cls_plan_storage[0..nl];
    try buildDetectBranchPlans(reg_plans, model_graph, weights_blob, reg_branch);
    try buildDetectBranchPlans(cls_plans, model_graph, weights_blob, cls_branch);
    const dfl_weights = if (reg_max > 1) blk: {
        var dfl_conv_buffer: [256]u8 = undefined;
        const dfl_conv_path = try utils.childModulePath(&dfl_conv_buffer, module_path, "dfl.conv");
        const dfl_spec = try spec.resolveConvSpec(model_graph, dfl_conv_path);
        break :blk weights_blob.slice(dfl_spec.weight);
    } else null;
    const score_logit_threshold = sigmoidThresholdToLogit(options.score_threshold);
    var profile = DetectProfile{};
    profile.level_count = nl;

    var candidates: std.ArrayListUnmanaged(Detection) = .empty;
    errdefer candidates.deinit(scratch_allocator);

    for (feature_inputs, 0..) |feature, level| {
        var reg_profile = DetectBranchProfile{};
        var timer = try std.time.Timer.start();
        var reg = try runDetectBranchPlanProfile(tensor_allocator, model_graph, weights_blob, reg_plans[level], feature, &reg_profile);
        profile.levels[level].reg_ns = timer.read();
        profile.levels[level].reg_detail = reg_profile;
        defer reg.deinit();
        var cls_profile = DetectBranchProfile{};
        timer.reset();
        var cls = try runDetectBranchPlanProfile(tensor_allocator, model_graph, weights_blob, cls_plans[level], feature, &cls_profile);
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
                    const best_score = sigmoid(best_logit);

                    const anchor_x = (@as(f32, @floatFromInt(x)) + 0.5) * stride;
                    const anchor_y = (@as(f32, @floatFromInt(y)) + 0.5) * stride;

                    const left = dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 0);
                    const top = dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 1);
                    const right = dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 2);
                    const bottom = dflExpectation(&reg, dfl_weights, reg_max, reg_batch_base, reg_plane, spatial_index, 3);

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
    const selected = try nms(scratch_allocator, output_allocator, candidates.items, options);
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

fn buildDetectBranchPlans(
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
                .conv1 = try buildConvPlan(model_graph, weights_blob, node.children[1].path),
                .head = try buildConvPlan(model_graph, weights_blob, node.children[2].path),
            } }
        else if (matchesDetectCv3Branch(node))
            .{ .cv3 = .{
                .stage0 = .{
                    .depthwise = try buildConvPlan(model_graph, weights_blob, node.children[0].children[0].path),
                    .pointwise = try buildConvPlan(model_graph, weights_blob, node.children[0].children[1].path),
                },
                .stage1 = .{
                    .depthwise = try buildConvPlan(model_graph, weights_blob, node.children[1].children[0].path),
                    .pointwise = try buildConvPlan(model_graph, weights_blob, node.children[1].children[1].path),
                },
                .head = try buildConvPlan(model_graph, weights_blob, node.children[2].path),
            } }
        else
            .{ .generic = node };
    }
}

fn buildConvPlan(
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
) !ConvPlan {
    const conv_spec = try spec.resolveConvSpec(model_graph, module_path);
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

fn runDetectBranchPlan(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    plan: BranchPlan,
    input: *const Tensor,
) !Tensor {
    return switch (plan) {
        .cv2 => |cv2| runDetectCv2BranchPlanned(allocator, &cv2, input),
        .cv3 => |cv3| runDetectCv3BranchPlanned(allocator, &cv3, input),
        .generic => |node| runNodeChain(allocator, model_graph, weights_blob, node, input),
    };
}

fn runDetectBranchPlanProfile(
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
            break :blk runDetectCv2BranchPlannedProfile(allocator, &cv2, input, profile);
        },
        .cv3 => |cv3| blk: {
            profile.kind = .cv3;
            break :blk runDetectCv3BranchPlannedProfile(allocator, &cv3, input, profile);
        },
        .generic => |node| blk: {
            profile.kind = .generic;
            break :blk runNodeChain(allocator, model_graph, weights_blob, node, input);
        },
    };
}

fn runNodeChain(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    node: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor {
    if (std.mem.eql(u8, node.kind, "Sequential")) {
        if (node.children.len == 0) return input.clone();

        var current = try runNodeChain(allocator, model_graph, weights_blob, &node.children[0], input);
        for (node.children[1..]) |*child| {
            const next = try runNodeChain(allocator, model_graph, weights_blob, child, &current);
            current.deinit();
            current = next;
        }
        return current;
    }

    return blocks.runModule(allocator, model_graph, weights_blob, node.path, input);
}

fn dflExpectation(
    reg: *const Tensor,
    dfl_weights: ?[]const f32,
    reg_max: usize,
    reg_batch_base: usize,
    reg_plane: usize,
    spatial_index: usize,
    side: usize,
) f32 {
    if (reg_max == 1) {
        return reg.data[reg_batch_base + side * reg_plane + spatial_index];
    }

    const weights = dfl_weights orelse unreachable;
    const channel_base = side * reg_max;
    var max_logit = reg.data[reg_batch_base + channel_base * reg_plane + spatial_index];
    for (1..reg_max) |bin| {
        const value = reg.data[reg_batch_base + (channel_base + bin) * reg_plane + spatial_index];
        if (value > max_logit) max_logit = value;
    }

    var denom: f32 = 0.0;
    var numer: f32 = 0.0;
    for (0..reg_max) |bin| {
        const prob = @exp(reg.data[reg_batch_base + (channel_base + bin) * reg_plane + spatial_index] - max_logit);
        denom += prob;
        numer += prob * weights[bin];
    }
    return numer / denom;
}

fn resolveDetectBranch(
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

fn matchesDetectCv2Branch(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 3 and
        std.mem.eql(u8, node.children[0].kind, "Conv") and
        std.mem.eql(u8, node.children[1].kind, "Conv") and
        std.mem.eql(u8, node.children[2].kind, "Conv2d");
}

fn matchesDetectCv3Stage(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 2 and
        std.mem.eql(u8, node.children[0].kind, "DWConv") and
        std.mem.eql(u8, node.children[1].kind, "Conv");
}

fn matchesDetectCv3Branch(node: *const graph.ModuleNode) bool {
    return std.mem.eql(u8, node.kind, "Sequential") and
        node.children.len == 3 and
        matchesDetectCv3Stage(&node.children[0]) and
        matchesDetectCv3Stage(&node.children[1]) and
        std.mem.eql(u8, node.children[2].kind, "Conv2d");
}

fn runDetectCv2Branch(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    node: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    var hidden0 = try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[0].path, input);
    defer hidden0.deinit();
    var hidden1 = try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[1].path, &hidden0);
    defer hidden1.deinit();
    return try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[2].path, &hidden1);
}

fn runDetectCv2BranchPlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv2BranchPlan,
    input: *const Tensor,
) !Tensor {
    var hidden0 = if (canUseFastDetectCv2Conv(input, &plan.conv0))
        try runDetectFast3x3Conv64Batch1(allocator, input, &plan.conv0)
    else
        try runConvPlan(allocator, &plan.conv0, input);
    defer hidden0.deinit();
    var hidden1 = if (canUseFastDetectCv2Conv(&hidden0, &plan.conv1))
        try runDetectFast3x3Conv64Batch1(allocator, &hidden0, &plan.conv1)
    else
        try runConvPlan(allocator, &plan.conv1, &hidden0);
    defer hidden1.deinit();
    return try runConvPlan(allocator, &plan.head, &hidden1);
}

fn runDetectCv2BranchPlannedProfile(
    allocator: std.mem.Allocator,
    plan: *const Cv2BranchPlan,
    input: *const Tensor,
    profile: *DetectBranchProfile,
) !Tensor {
    var timer = try std.time.Timer.start();
    var hidden0 = if (canUseFastDetectCv2Conv(input, &plan.conv0))
        try runDetectFast3x3Conv64Batch1(allocator, input, &plan.conv0)
    else
        try runConvPlan(allocator, &plan.conv0, input);
    profile.stage0_ns = timer.read();
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = if (canUseFastDetectCv2Conv(&hidden0, &plan.conv1))
        try runDetectFast3x3Conv64Batch1(allocator, &hidden0, &plan.conv1)
    else
        try runConvPlan(allocator, &plan.conv1, &hidden0);
    profile.stage1_ns = timer.read();
    defer hidden1.deinit();

    timer.reset();
    const output = try runConvPlan(allocator, &plan.head, &hidden1);
    profile.stage2_ns = timer.read();
    return output;
}

fn runDetectCv3Stage(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    node: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    var hidden = try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[0].path, input);
    defer hidden.deinit();
    return try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[1].path, &hidden);
}

fn runDetectCv3StagePlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv3StagePlan,
    input: *const Tensor,
) !Tensor {
    var hidden = try runConvPlan(allocator, &plan.depthwise, input);
    defer hidden.deinit();
    return try runConvPlan(allocator, &plan.pointwise, &hidden);
}

fn runDetectCv3Branch(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    node: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    var hidden0 = try runDetectCv3Stage(allocator, model_graph, weights_blob, &node.children[0], input);
    defer hidden0.deinit();
    var hidden1 = try runDetectCv3Stage(allocator, model_graph, weights_blob, &node.children[1], &hidden0);
    defer hidden1.deinit();
    return try blocks.runConvModule(allocator, model_graph, weights_blob, node.children[2].path, &hidden1);
}

fn runDetectCv3BranchPlanned(
    allocator: std.mem.Allocator,
    plan: *const Cv3BranchPlan,
    input: *const Tensor,
) !Tensor {
    var hidden0 = try runDetectCv3StagePlanned(allocator, &plan.stage0, input);
    defer hidden0.deinit();
    var hidden1 = try runDetectCv3StagePlanned(allocator, &plan.stage1, &hidden0);
    defer hidden1.deinit();
    return try runConvPlan(allocator, &plan.head, &hidden1);
}

fn runDetectCv3BranchPlannedProfile(
    allocator: std.mem.Allocator,
    plan: *const Cv3BranchPlan,
    input: *const Tensor,
    profile: *DetectBranchProfile,
) !Tensor {
    var timer = try std.time.Timer.start();
    var hidden0 = try runConvPlan(allocator, &plan.stage0.depthwise, input);
    profile.stage0_ns = timer.read();
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = try runConvPlan(allocator, &plan.stage0.pointwise, &hidden0);
    profile.stage1_ns = timer.read();
    defer hidden1.deinit();

    timer.reset();
    var hidden2 = try runConvPlan(allocator, &plan.stage1.depthwise, &hidden1);
    profile.stage2_ns = timer.read();
    defer hidden2.deinit();

    timer.reset();
    var hidden3 = try runConvPlan(allocator, &plan.stage1.pointwise, &hidden2);
    profile.stage3_ns = timer.read();
    defer hidden3.deinit();

    timer.reset();
    const output = try runConvPlan(allocator, &plan.head, &hidden3);
    profile.stage4_ns = timer.read();
    return output;
}

fn runConvPlan(
    allocator: std.mem.Allocator,
    plan: *const ConvPlan,
    input: *const Tensor,
) !Tensor {
    const out_height = ((input.shape[2] + 2 * plan.pad_h - plan.weight.shape[2]) / plan.stride_h) + 1;
    const out_width = ((input.shape[3] + 2 * plan.pad_w - plan.weight.shape[3]) / plan.stride_w) + 1;

    var output = try Tensor.init(allocator, input.shape[0], plan.weight.shape[0], out_height, out_width);
    errdefer output.deinit();

    try ops.conv2d(input, &plan.weight, plan.bias, &output, .{
        .stride_h = plan.stride_h,
        .stride_w = plan.stride_w,
        .pad_h = plan.pad_h,
        .pad_w = plan.pad_w,
        .groups = plan.groups,
        .apply_silu = plan.activation == .silu,
    });
    if (plan.activation != .silu) {
        utils.applyActivation(&output, plan.activation);
    }
    return output;
}

fn canUseFastDetectCv2Conv(input: *const Tensor, plan: *const ConvPlan) bool {
    return input.shape[0] == 1 and
        plan.weight.shape[0] == 64 and
        plan.weight.shape[2] == 3 and
        plan.weight.shape[3] == 3 and
        plan.stride_h == 1 and
        plan.stride_w == 1 and
        plan.pad_h == 1 and
        plan.pad_w == 1 and
        plan.groups == 1 and
        plan.activation == .silu;
}


fn runDetectFast3x3Conv64Batch1(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
) !Tensor {
    const thread_count = chooseDetectFastConvThreadCount(input.shape[2] * input.shape[3]);
    if (thread_count > 1) {
        return runDetectFast3x3Conv64Batch1Parallel(allocator, input, plan, thread_count);
    }
    return runDetectFast3x3Conv64Batch1Range(allocator, input, plan, 0, 64);
}

fn runDetectFast3x3Conv64Batch1Parallel(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    thread_count: usize,
) !Tensor {
    var output = try Tensor.init(allocator, 1, 64, input.shape[2], input.shape[3]);
    errdefer output.deinit();

    var threads: [max_detect_fast_threads - 1]std.Thread = undefined;
    var spawned: usize = 0;

    for (0..thread_count) |thread_index| {
        const oc_start = (64 * thread_index) / thread_count;
        const oc_end = (64 * (thread_index + 1)) / thread_count;
        if (oc_start == oc_end) continue;

        if (thread_index + 1 == thread_count) {
            runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
        } else {
            threads[spawned] = std.Thread.spawn(.{}, runDetectFast3x3Conv64Batch1Worker, .{
                DetectFastConvTask{
                    .input = input,
                    .plan = plan,
                    .output = &output,
                    .oc_start = oc_start,
                    .oc_end = oc_end,
                },
            }) catch {
                runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
                continue;
            };
            spawned += 1;
        }
    }

    for (threads[0..spawned]) |thread| thread.join();
    return output;
}

fn runDetectFast3x3Conv64Batch1Range(
    allocator: std.mem.Allocator,
    input: *const Tensor,
    plan: *const ConvPlan,
    oc_start: usize,
    oc_end: usize,
) !Tensor {
    const out_height = input.shape[2];
    const out_width = input.shape[3];

    var output = try Tensor.init(allocator, 1, 64, out_height, out_width);
    errdefer output.deinit();
    runDetectFast3x3Conv64Batch1Into(input, plan, &output, oc_start, oc_end);
    return output;
}

fn runDetectFast3x3Conv64Batch1Into(
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    oc_start: usize,
    oc_end: usize,
) void {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const out_height = input.shape[2];
    const out_width = input.shape[3];
    const input_plane = in_height * in_width;
    const output_plane = out_height * out_width;
    const interior_h_end = if (out_height > 1) out_height - 1 else 0;
    const interior_w_end = if (out_width > 1) out_width - 1 else 0;

    var oc: usize = oc_start;
    while (oc + 1 < oc_end) : (oc += 2) {
        const weight0_base = oc * in_channels * 9;
        const weight1_base = (oc + 1) * in_channels * 9;
        const out0_base = oc * output_plane;
        const out1_base = (oc + 1) * output_plane;
        const bias0: f32 = if (plan.bias) |b| b[oc] else 0.0;
        const bias1: f32 = if (plan.bias) |b| b[oc + 1] else 0.0;

        for (0..out_height) |oy| {
            const out0_row = out0_base + oy * out_width;
            const out1_row = out1_base + oy * out_width;

            if (oy == 0 or oy >= interior_h_end) {
                for (0..out_width) |ox| {
                    const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, ox);
                    output.data[out0_row + ox] = silu(acc.a);
                    output.data[out1_row + ox] = silu(acc.b);
                }
                continue;
            }

            {
                const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, 0);
                output.data[out0_row] = silu(acc.a);
                output.data[out1_row] = silu(acc.b);
            }

            for (1..interior_w_end) |ox| {
                var acc0: f32 = bias0;
                var acc1: f32 = bias1;
                const row0 = (oy - 1) * in_width + (ox - 1);
                const row1 = row0 + in_width;
                const row2 = row1 + in_width;

                for (0..in_channels) |ic| {
                    const input_channel = input.data[ic * input_plane ..][0..input_plane];
                    const ic_weight0 = weight0_base + ic * 9;
                    const ic_weight1 = weight1_base + ic * 9;

                    const v00 = input_channel[row0];
                    const v01 = input_channel[row0 + 1];
                    const v02 = input_channel[row0 + 2];
                    const v10 = input_channel[row1];
                    const v11 = input_channel[row1 + 1];
                    const v12 = input_channel[row1 + 2];
                    const v20 = input_channel[row2];
                    const v21 = input_channel[row2 + 1];
                    const v22 = input_channel[row2 + 2];

                    acc0 += v00 * plan.weight.data[ic_weight0];
                    acc0 += v01 * plan.weight.data[ic_weight0 + 1];
                    acc0 += v02 * plan.weight.data[ic_weight0 + 2];
                    acc0 += v10 * plan.weight.data[ic_weight0 + 3];
                    acc0 += v11 * plan.weight.data[ic_weight0 + 4];
                    acc0 += v12 * plan.weight.data[ic_weight0 + 5];
                    acc0 += v20 * plan.weight.data[ic_weight0 + 6];
                    acc0 += v21 * plan.weight.data[ic_weight0 + 7];
                    acc0 += v22 * plan.weight.data[ic_weight0 + 8];

                    acc1 += v00 * plan.weight.data[ic_weight1];
                    acc1 += v01 * plan.weight.data[ic_weight1 + 1];
                    acc1 += v02 * plan.weight.data[ic_weight1 + 2];
                    acc1 += v10 * plan.weight.data[ic_weight1 + 3];
                    acc1 += v11 * plan.weight.data[ic_weight1 + 4];
                    acc1 += v12 * plan.weight.data[ic_weight1 + 5];
                    acc1 += v20 * plan.weight.data[ic_weight1 + 6];
                    acc1 += v21 * plan.weight.data[ic_weight1 + 7];
                    acc1 += v22 * plan.weight.data[ic_weight1 + 8];
                }

                output.data[out0_row + ox] = silu(acc0);
                output.data[out1_row + ox] = silu(acc1);
            }

            for (interior_w_end..out_width) |ox| {
                const acc = detectFast3x3PointPair(input, &plan.weight, bias0, bias1, weight0_base, weight1_base, oy, ox);
                output.data[out0_row + ox] = silu(acc.a);
                output.data[out1_row + ox] = silu(acc.b);
            }
        }
    }

    while (oc < oc_end) : (oc += 1) {
        const weight_base = oc * in_channels * 9;
        const out_base = oc * output_plane;
        const bias_value: f32 = if (plan.bias) |b| b[oc] else 0.0;

        for (0..out_height) |oy| {
            const out_row = out_base + oy * out_width;
            if (oy == 0 or oy >= interior_h_end) {
                for (0..out_width) |ox| {
                    output.data[out_row + ox] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, ox));
                }
                continue;
            }

            output.data[out_row] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, 0));

            for (1..interior_w_end) |ox| {
                var acc: f32 = bias_value;
                const row0 = (oy - 1) * in_width + (ox - 1);
                const row1 = row0 + in_width;
                const row2 = row1 + in_width;

                for (0..in_channels) |ic| {
                    const input_channel = input.data[ic * input_plane ..][0..input_plane];
                    const ic_weight = weight_base + ic * 9;
                    acc += input_channel[row0] * plan.weight.data[ic_weight];
                    acc += input_channel[row0 + 1] * plan.weight.data[ic_weight + 1];
                    acc += input_channel[row0 + 2] * plan.weight.data[ic_weight + 2];
                    acc += input_channel[row1] * plan.weight.data[ic_weight + 3];
                    acc += input_channel[row1 + 1] * plan.weight.data[ic_weight + 4];
                    acc += input_channel[row1 + 2] * plan.weight.data[ic_weight + 5];
                    acc += input_channel[row2] * plan.weight.data[ic_weight + 6];
                    acc += input_channel[row2 + 1] * plan.weight.data[ic_weight + 7];
                    acc += input_channel[row2 + 2] * plan.weight.data[ic_weight + 8];
                }

                output.data[out_row + ox] = silu(acc);
            }

            for (interior_w_end..out_width) |ox| {
                output.data[out_row + ox] = silu(detectFast3x3Point(input, &plan.weight, bias_value, weight_base, oy, ox));
            }
        }
    }

}

const DetectFastConvTask = struct {
    input: *const Tensor,
    plan: *const ConvPlan,
    output: *Tensor,
    oc_start: usize,
    oc_end: usize,
};

fn runDetectFast3x3Conv64Batch1Worker(task: DetectFastConvTask) void {
    runDetectFast3x3Conv64Batch1Into(task.input, task.plan, task.output, task.oc_start, task.oc_end);
}

fn chooseDetectFastConvThreadCount(spatial: usize) usize {
    if (spatial >= 128) return 2;
    return 1;
}

const DetectPairAcc = struct {
    a: f32,
    b: f32,
};

fn detectFast3x3PointPair(
    input: *const Tensor,
    weights: *const Tensor,
    bias0: f32,
    bias1: f32,
    weight0_base: usize,
    weight1_base: usize,
    oy: usize,
    ox: usize,
) DetectPairAcc {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const input_plane = in_height * in_width;
    const base_y = @as(isize, @intCast(oy)) - 1;
    const base_x = @as(isize, @intCast(ox)) - 1;

    var acc0: f32 = bias0;
    var acc1: f32 = bias1;
    for (0..in_channels) |ic| {
        const input_channel = input.data[ic * input_plane ..][0..input_plane];
        const ic_weight0 = weight0_base + ic * 9;
        const ic_weight1 = weight1_base + ic * 9;
        const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
        const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
        const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
        const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

        var iy = y_start;
        while (iy < y_end) : (iy += 1) {
            const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
            const input_row_base = iy * in_width;
            const weight0_row = ic_weight0 + ky * 3;
            const weight1_row = ic_weight1 + ky * 3;
            var ix = x_start;
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                const src = input_channel[input_row_base + ix];
                acc0 += src * weights.data[weight0_row + kx];
                acc1 += src * weights.data[weight1_row + kx];
            }
        }
    }
    return .{ .a = acc0, .b = acc1 };
}

fn detectFast3x3Point(
    input: *const Tensor,
    weights: *const Tensor,
    bias_value: f32,
    weight_base: usize,
    oy: usize,
    ox: usize,
) f32 {
    const in_channels = input.shape[1];
    const in_height = input.shape[2];
    const in_width = input.shape[3];
    const input_plane = in_height * in_width;
    const base_y = @as(isize, @intCast(oy)) - 1;
    const base_x = @as(isize, @intCast(ox)) - 1;

    var acc: f32 = bias_value;
    for (0..in_channels) |ic| {
        const input_channel = input.data[ic * input_plane ..][0..input_plane];
        const ic_weight = weight_base + ic * 9;
        const y_start: usize = @intCast(@max(@as(isize, 0), base_y));
        const y_end: usize = @intCast(@min(@as(isize, @intCast(in_height)), base_y + 3));
        const x_start: usize = @intCast(@max(@as(isize, 0), base_x));
        const x_end: usize = @intCast(@min(@as(isize, @intCast(in_width)), base_x + 3));

        var iy = y_start;
        while (iy < y_end) : (iy += 1) {
            const ky = @as(usize, @intCast(@as(isize, @intCast(iy)) - base_y));
            const input_row_base = iy * in_width;
            const weight_row = ic_weight + ky * 3;
            var ix = x_start;
            while (ix < x_end) : (ix += 1) {
                const kx = @as(usize, @intCast(@as(isize, @intCast(ix)) - base_x));
                acc += input_channel[input_row_base + ix] * weights.data[weight_row + kx];
            }
        }
    }
    return acc;
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn sigmoid(value: f32) f32 {
    return 1.0 / (1.0 + @exp(-value));
}

fn sigmoidThresholdToLogit(threshold: f32) f32 {
    if (threshold <= 0.0) return -std.math.inf(f32);
    if (threshold >= 1.0) return std.math.inf(f32);
    return @log(threshold / (1.0 - threshold));
}

fn nms(
    scratch_allocator: std.mem.Allocator,
    output_allocator: std.mem.Allocator,
    detections: []const Detection,
    options: DetectOptions,
) ![]Detection {
    var states = try scratch_allocator.alloc(u8, detections.len);
    defer scratch_allocator.free(states);
    @memset(states, 0);

    var selected: std.ArrayListUnmanaged(Detection) = .empty;
    errdefer selected.deinit(output_allocator);

    while (selected.items.len < options.max_det) {
        var best_index: ?usize = null;
        var best_score: f32 = -1.0;

        for (detections, 0..) |det, index| {
            if (states[index] != 0) continue;
            if (det.score > best_score) {
                best_score = det.score;
                best_index = index;
            }
        }

        const winner = best_index orelse break;
        states[winner] = 2;
        try selected.append(output_allocator, detections[winner]);

        for (detections, 0..) |det, index| {
            if (states[index] != 0) continue;
            if (det.class_id != detections[winner].class_id) continue;
            if (iou(det, detections[winner]) > options.iou_threshold) {
                states[index] = 1;
            }
        }
    }

    return try selected.toOwnedSlice(output_allocator);
}

fn iou(lhs: Detection, rhs: Detection) f32 {
    const inter_x1 = @max(lhs.x1, rhs.x1);
    const inter_y1 = @max(lhs.y1, rhs.y1);
    const inter_x2 = @min(lhs.x2, rhs.x2);
    const inter_y2 = @min(lhs.y2, rhs.y2);

    const inter_w = @max(@as(f32, 0.0), inter_x2 - inter_x1);
    const inter_h = @max(@as(f32, 0.0), inter_y2 - inter_y1);
    const inter_area = inter_w * inter_h;
    if (inter_area <= 0.0) return 0.0;

    const lhs_area = @max(@as(f32, 0.0), lhs.x2 - lhs.x1) * @max(@as(f32, 0.0), lhs.y2 - lhs.y1);
    const rhs_area = @max(@as(f32, 0.0), rhs.x2 - rhs.x1) * @max(@as(f32, 0.0), rhs.y2 - rhs.y1);
    const union_area = lhs_area + rhs_area - inter_area;
    if (union_area <= 0.0) return 0.0;
    return inter_area / union_area;
}
