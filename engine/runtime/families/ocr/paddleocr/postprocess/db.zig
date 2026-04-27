const std = @import("std");

pub const Options = struct {
    threshold: f32 = 0.3,
    min_area: usize = 1,
    min_score: f32 = 0.0,
    expand_pixels: usize = 1,
    unclip_ratio: f32 = 1.5,
    nms_threshold: f32 = 0.3,
    sort_reading_order: bool = true,
};

pub const Point = struct {
    x: f32,
    y: f32,
};

pub const Box = struct {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    area: usize,
    score: f32,
    points: [4]Point = undefined,
};

const Component = struct {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    area: usize,
    score_sum: f32,
};

pub fn boxesFromProbabilityMap(
    allocator: std.mem.Allocator,
    map: []const f32,
    width: usize,
    height: usize,
    options: Options,
) ![]Box {
    if (width == 0 or height == 0 or map.len != width * height) return error.ShapeMismatch;

    const visited = try allocator.alloc(bool, map.len);
    defer allocator.free(visited);
    @memset(visited, false);

    var stack = std.ArrayListUnmanaged(usize).empty;
    defer stack.deinit(allocator);
    var boxes = std.ArrayListUnmanaged(Box).empty;
    errdefer boxes.deinit(allocator);

    for (map, 0..) |value, index| {
        if (visited[index] or value < options.threshold) continue;
        const component = try floodFill(allocator, map, width, height, options.threshold, visited, &stack, index);
        if (component.area < options.min_area) continue;
        const score = component.score_sum / @as(f32, @floatFromInt(component.area));
        if (score < options.min_score) continue;
        try boxes.append(allocator, expandBox(.{
            .x_min = component.x_min,
            .y_min = component.y_min,
            .x_max = component.x_max,
            .y_max = component.y_max,
            .area = component.area,
            .score = score,
        }, width, height, options.expand_pixels, options.unclip_ratio));
    }

    if (options.sort_reading_order) sortBoxesReadingOrder(boxes.items);
    if (options.nms_threshold > 0 and boxes.items.len > 1) try applyNms(allocator, &boxes, options.nms_threshold);
    return try boxes.toOwnedSlice(allocator);
}

fn expandBox(box: Box, width: usize, height: usize, pixels: usize, unclip_ratio: f32) Box {
    const box_w = box.x_max - box.x_min + 1;
    const box_h = box.y_max - box.y_min + 1;
    const ratio_pad: usize = if (unclip_ratio <= 1.0)
        0
    else
        @intFromFloat(@ceil(@as(f32, @floatFromInt(@max(box_w, box_h))) * (unclip_ratio - 1.0) * 0.5));
    const pad = pixels + ratio_pad;
    return boxWithPoints(.{
        .x_min = if (box.x_min > pad) box.x_min - pad else 0,
        .y_min = if (box.y_min > pad) box.y_min - pad else 0,
        .x_max = @min(width - 1, box.x_max + pad),
        .y_max = @min(height - 1, box.y_max + pad),
        .area = box.area,
        .score = box.score,
    });
}

pub fn boxWithPoints(box: Box) Box {
    var out = box;
    out.points = .{
        .{ .x = @floatFromInt(box.x_min), .y = @floatFromInt(box.y_min) },
        .{ .x = @floatFromInt(box.x_max), .y = @floatFromInt(box.y_min) },
        .{ .x = @floatFromInt(box.x_max), .y = @floatFromInt(box.y_max) },
        .{ .x = @floatFromInt(box.x_min), .y = @floatFromInt(box.y_max) },
    };
    return out;
}

fn sortBoxesReadingOrder(boxes: []Box) void {
    std.mem.sort(Box, boxes, {}, struct {
        fn lessThan(_: void, lhs: Box, rhs: Box) bool {
            const lhs_mid_y = lhs.y_min + (lhs.y_max - lhs.y_min) / 2;
            const rhs_mid_y = rhs.y_min + (rhs.y_max - rhs.y_min) / 2;
            if (lhs_mid_y != rhs_mid_y) return lhs_mid_y < rhs_mid_y;
            return lhs.x_min < rhs.x_min;
        }
    }.lessThan);
}

fn applyNms(allocator: std.mem.Allocator, boxes: *std.ArrayListUnmanaged(Box), threshold: f32) !void {
    std.mem.sort(Box, boxes.items, {}, struct {
        fn lessThan(_: void, lhs: Box, rhs: Box) bool {
            return lhs.score > rhs.score;
        }
    }.lessThan);

    const keep = try allocator.alloc(bool, boxes.items.len);
    defer allocator.free(keep);
    @memset(keep, true);
    for (boxes.items, 0..) |candidate, index| {
        if (!keep[index]) continue;
        for (boxes.items[index + 1 ..], index + 1..) |other, other_index| {
            if (!keep[other_index]) continue;
            if (boxIou(candidate, other) > threshold) keep[other_index] = false;
        }
    }

    var write_index: usize = 0;
    for (boxes.items, keep) |box, should_keep| {
        if (!should_keep) continue;
        boxes.items[write_index] = box;
        write_index += 1;
    }
    boxes.items.len = write_index;
    if (boxes.items.len > 1) sortBoxesReadingOrder(boxes.items);
}

fn boxIou(a: Box, b: Box) f32 {
    const x0 = @max(a.x_min, b.x_min);
    const y0 = @max(a.y_min, b.y_min);
    const x1 = @min(a.x_max, b.x_max);
    const y1 = @min(a.y_max, b.y_max);
    if (x1 < x0 or y1 < y0) return 0;
    const intersection = (x1 - x0 + 1) * (y1 - y0 + 1);
    const a_area = (a.x_max - a.x_min + 1) * (a.y_max - a.y_min + 1);
    const b_area = (b.x_max - b.x_min + 1) * (b.y_max - b.y_min + 1);
    const union_area = a_area + b_area - intersection;
    if (union_area == 0) return 0;
    return @as(f32, @floatFromInt(intersection)) / @as(f32, @floatFromInt(union_area));
}

fn floodFill(
    allocator: std.mem.Allocator,
    map: []const f32,
    width: usize,
    height: usize,
    threshold: f32,
    visited: []bool,
    stack: *std.ArrayListUnmanaged(usize),
    start: usize,
) !Component {
    stack.clearRetainingCapacity();
    try stack.append(allocator, start);
    visited[start] = true;

    var component = Component{
        .x_min = start % width,
        .y_min = start / width,
        .x_max = start % width,
        .y_max = start / width,
        .area = 0,
        .score_sum = 0,
    };

    while (stack.items.len != 0) {
        const index = stack.items[stack.items.len - 1];
        stack.items.len -= 1;
        const x = index % width;
        const y = index / width;
        const score = map[index];

        component.area += 1;
        component.score_sum += score;
        component.x_min = @min(component.x_min, x);
        component.y_min = @min(component.y_min, y);
        component.x_max = @max(component.x_max, x);
        component.y_max = @max(component.y_max, y);

        try pushNeighbor(allocator, map, width, height, threshold, visited, stack, x, y, -1, 0);
        try pushNeighbor(allocator, map, width, height, threshold, visited, stack, x, y, 1, 0);
        try pushNeighbor(allocator, map, width, height, threshold, visited, stack, x, y, 0, -1);
        try pushNeighbor(allocator, map, width, height, threshold, visited, stack, x, y, 0, 1);
    }

    return component;
}

fn pushNeighbor(
    allocator: std.mem.Allocator,
    map: []const f32,
    width: usize,
    height: usize,
    threshold: f32,
    visited: []bool,
    stack: *std.ArrayListUnmanaged(usize),
    x: usize,
    y: usize,
    dx: isize,
    dy: isize,
) !void {
    const nx_signed: isize = @as(isize, @intCast(x)) + dx;
    const ny_signed: isize = @as(isize, @intCast(y)) + dy;
    if (nx_signed < 0 or ny_signed < 0) return;
    const nx: usize = @intCast(nx_signed);
    const ny: usize = @intCast(ny_signed);
    if (nx >= width or ny >= height) return;

    const index = ny * width + nx;
    if (visited[index]) return;
    visited[index] = true;
    if (map[index] < threshold) return;
    try stack.append(allocator, index);
}

test "paddleocr db postprocess extracts connected box" {
    const boxes = try boxesFromProbabilityMap(
        std.testing.allocator,
        &.{
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.8, 0.7, 0.0,
            0.0, 0.9, 0.0, 0.0,
        },
        4,
        3,
        .{ .threshold = 0.5, .min_area = 2, .expand_pixels = 0, .unclip_ratio = 1.0 },
    );
    defer std.testing.allocator.free(boxes);

    try std.testing.expectEqual(@as(usize, 1), boxes.len);
    try std.testing.expectEqual(@as(usize, 1), boxes[0].x_min);
    try std.testing.expectEqual(@as(usize, 1), boxes[0].y_min);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].x_max);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].y_max);
    try std.testing.expectEqual(@as(usize, 3), boxes[0].area);
    try std.testing.expect(boxes[0].score > 0.79 and boxes[0].score < 0.81);
}

test "paddleocr db postprocess filters expands and sorts boxes" {
    const boxes = try boxesFromProbabilityMap(
        std.testing.allocator,
        &.{
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.9, 0.0, 0.8,
            0.0, 0.0, 0.0, 0.0,
            0.7, 0.0, 0.0, 0.0,
        },
        4,
        4,
        .{ .threshold = 0.5, .min_score = 0.75, .expand_pixels = 1, .unclip_ratio = 1.0 },
    );
    defer std.testing.allocator.free(boxes);

    try std.testing.expectEqual(@as(usize, 2), boxes.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[0].x_min);
    try std.testing.expectEqual(@as(usize, 0), boxes[0].y_min);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].x_max);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].y_max);
    try std.testing.expectEqual(@as(usize, 2), boxes[1].x_min);
}

test "paddleocr db postprocess emits quad points and suppresses overlaps" {
    const boxes = try boxesFromProbabilityMap(
        std.testing.allocator,
        &.{
            0.9, 0.8, 0.0,
            0.7, 0.6, 0.0,
            0.0, 0.0, 0.9,
        },
        3,
        3,
        .{ .threshold = 0.5, .expand_pixels = 0, .unclip_ratio = 1.5, .nms_threshold = 0.1 },
    );
    defer std.testing.allocator.free(boxes);

    try std.testing.expect(boxes.len >= 1);
    try std.testing.expectEqual(@as(f32, 0), boxes[0].points[0].x);
    try std.testing.expectEqual(@as(f32, 0), boxes[0].points[0].y);
}
