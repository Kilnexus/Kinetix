const std = @import("std");
const graph = @import("graph");
const types = @import("engine_vision_base").types;

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

pub const max_detect_branch_levels = 8;
pub const max_detect_fast_threads = 2;

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
    candidate_count: usize = 0,
    max_class_logit: f32 = 0.0,
    max_class_score: f32 = 0.0,
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

pub const ConvPlan = struct {
    weight: Tensor,
    bias: ?[]const f32,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    groups: usize,
    activation: types.Activation,
};

pub const Cv2BranchPlan = struct {
    conv0: ConvPlan,
    conv0_fast: bool = false,
    conv1: ConvPlan,
    conv1_fast: bool = false,
    head: ConvPlan,
};

pub const Cv3StagePlan = struct {
    depthwise: ConvPlan,
    depthwise_fast: bool = false,
    pointwise: ConvPlan,
};

pub const Cv3BranchPlan = struct {
    stage0: Cv3StagePlan,
    stage1: Cv3StagePlan,
    head: ConvPlan,
};

pub const BranchPlan = union(enum) {
    cv2: Cv2BranchPlan,
    cv3: Cv3BranchPlan,
    generic: *const graph.ModuleNode,
};

pub const CachedDetectPlan = struct {
    nl: usize,
    nc: usize,
    reg_max: usize,
    dfl_weights: ?[]const f32,
    reg_plans: [max_detect_branch_levels]BranchPlan,
    cls_plans: [max_detect_branch_levels]BranchPlan,
};
