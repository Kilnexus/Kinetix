const std = @import("std");
const graph = @import("graph");
const execute = @import("execute.zig");
const spec = @import("spec.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
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

pub fn runDetect(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !DetectOutput {
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

    var dfl_conv_buffer: [256]u8 = undefined;
    const dfl_conv_path = try utils.childModulePath(&dfl_conv_buffer, module_path, "dfl.conv");
    const dfl_spec = try spec.resolveConvSpec(model_graph, dfl_conv_path);
    const dfl_weights = weights_blob.slice(dfl_spec.weight);

    var candidates: std.ArrayListUnmanaged(Detection) = .empty;
    errdefer candidates.deinit(allocator);

    for (feature_inputs, 0..) |feature, level| {
        var reg = try runDetectBranch(allocator, model_graph, weights_blob, module_path, "cv2", level, feature);
        defer reg.deinit();
        var cls = try runDetectBranch(allocator, model_graph, weights_blob, module_path, "cv3", level, feature);
        defer cls.deinit();

        if (reg.shape[0] != cls.shape[0] or reg.shape[2] != cls.shape[2] or reg.shape[3] != cls.shape[3]) {
            return error.InvalidAttributeType;
        }

        const stride = model_graph.strides[level];
        for (0..reg.shape[0]) |n| {
            for (0..reg.shape[2]) |y| {
                for (0..reg.shape[3]) |x| {
                    var best_score: f32 = 0.0;
                    var best_class: usize = 0;
                    for (0..nc) |class_idx| {
                        const score = sigmoid(cls.get(n, class_idx, y, x));
                        if (score > best_score) {
                            best_score = score;
                            best_class = class_idx;
                        }
                    }
                    if (best_score < options.score_threshold) continue;

                    const anchor_x = (@as(f32, @floatFromInt(x)) + 0.5) * stride;
                    const anchor_y = (@as(f32, @floatFromInt(y)) + 0.5) * stride;

                    const left = dflExpectation(&reg, dfl_weights, reg_max, n, 0, y, x);
                    const top = dflExpectation(&reg, dfl_weights, reg_max, n, 1, y, x);
                    const right = dflExpectation(&reg, dfl_weights, reg_max, n, 2, y, x);
                    const bottom = dflExpectation(&reg, dfl_weights, reg_max, n, 3, y, x);

                    try candidates.append(allocator, .{
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
    }

    const candidate_count = candidates.items.len;
    const selected = try nms(allocator, candidates.items, options);
    candidates.deinit(allocator);

    return .{
        .allocator = allocator,
        .detections = selected,
        .candidate_count = candidate_count,
    };
}

fn runDetectBranch(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    detect_path: []const u8,
    branch_name: []const u8,
    branch_index: usize,
    input: *const Tensor,
) !Tensor {
    var branch_path_buffer: [256]u8 = undefined;
    const branch_path = try utils.childModulePath(&branch_path_buffer, detect_path, branch_name);
    const branch = model_graph.findModule(branch_path) orelse return error.ModuleNotFound;
    if (branch_index >= branch.children.len) return error.InvalidAttributeType;
    return runNodeChain(allocator, model_graph, weights_blob, &branch.children[branch_index], input);
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

    return execute.runModule(allocator, model_graph, weights_blob, node.path, input);
}

fn dflExpectation(
    reg: *const Tensor,
    dfl_weights: []const f32,
    reg_max: usize,
    batch: usize,
    side: usize,
    y: usize,
    x: usize,
) f32 {
    const channel_base = side * reg_max;
    var max_logit = reg.get(batch, channel_base, y, x);
    for (1..reg_max) |bin| {
        const value = reg.get(batch, channel_base + bin, y, x);
        if (value > max_logit) max_logit = value;
    }

    var denom: f32 = 0.0;
    var numer: f32 = 0.0;
    for (0..reg_max) |bin| {
        const prob = @exp(reg.get(batch, channel_base + bin, y, x) - max_logit);
        denom += prob;
        numer += prob * dfl_weights[bin];
    }
    return numer / denom;
}

fn sigmoid(value: f32) f32 {
    return 1.0 / (1.0 + @exp(-value));
}

fn nms(
    allocator: std.mem.Allocator,
    detections: []const Detection,
    options: DetectOptions,
) ![]Detection {
    var states = try allocator.alloc(u8, detections.len);
    defer allocator.free(states);
    @memset(states, 0);

    var selected: std.ArrayListUnmanaged(Detection) = .empty;
    errdefer selected.deinit(allocator);

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
        try selected.append(allocator, detections[winner]);

        for (detections, 0..) |det, index| {
            if (states[index] != 0) continue;
            if (det.class_id != detections[winner].class_id) continue;
            if (iou(det, detections[winner]) > options.iou_threshold) {
                states[index] = 1;
            }
        }
    }

    return try selected.toOwnedSlice(allocator);
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
