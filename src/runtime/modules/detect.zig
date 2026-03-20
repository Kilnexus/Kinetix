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
    var hidden0 = try runConvPlan(allocator, &plan.conv0, input);
    defer hidden0.deinit();
    var hidden1 = try runConvPlan(allocator, &plan.conv1, &hidden0);
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
    var hidden0 = try runConvPlan(allocator, &plan.conv0, input);
    profile.stage0_ns = timer.read();
    defer hidden0.deinit();

    timer.reset();
    var hidden1 = try runConvPlan(allocator, &plan.conv1, &hidden0);
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
    });
    utils.applyActivation(&output, plan.activation);
    return output;
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
