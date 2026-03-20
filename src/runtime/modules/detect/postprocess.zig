const std = @import("std");
const detect_types = @import("types.zig");

const Tensor = detect_types.Tensor;
const Detection = detect_types.Detection;
const DetectOptions = detect_types.DetectOptions;

pub fn dflExpectation(
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

pub fn sigmoid(value: f32) f32 {
    return 1.0 / (1.0 + @exp(-value));
}

pub fn sigmoidThresholdToLogit(threshold: f32) f32 {
    if (threshold <= 0.0) return -std.math.inf(f32);
    if (threshold >= 1.0) return std.math.inf(f32);
    return @log(threshold / (1.0 - threshold));
}

pub fn nms(
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

pub fn iou(lhs: Detection, rhs: Detection) f32 {
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
