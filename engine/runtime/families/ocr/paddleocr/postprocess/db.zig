const std = @import("std");

pub const Options = struct {
    threshold: f32 = 0.3,
    min_area: usize = 1,
    min_score: f32 = 0.0,
    expand_pixels: usize = 1,
    sort_reading_order: bool = true,
};

pub const Box = struct {
    x_min: usize,
    y_min: usize,
    x_max: usize,
    y_max: usize,
    area: usize,
    score: f32,
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
        }, width, height, options.expand_pixels));
    }

    if (options.sort_reading_order) sortBoxesReadingOrder(boxes.items);
    return try boxes.toOwnedSlice(allocator);
}

fn expandBox(box: Box, width: usize, height: usize, pixels: usize) Box {
    if (pixels == 0) return box;
    return .{
        .x_min = if (box.x_min > pixels) box.x_min - pixels else 0,
        .y_min = if (box.y_min > pixels) box.y_min - pixels else 0,
        .x_max = @min(width - 1, box.x_max + pixels),
        .y_max = @min(height - 1, box.y_max + pixels),
        .area = box.area,
        .score = box.score,
    };
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
        .{ .threshold = 0.5, .min_area = 2, .expand_pixels = 0 },
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
        .{ .threshold = 0.5, .min_score = 0.75, .expand_pixels = 1 },
    );
    defer std.testing.allocator.free(boxes);

    try std.testing.expectEqual(@as(usize, 2), boxes.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[0].x_min);
    try std.testing.expectEqual(@as(usize, 0), boxes[0].y_min);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].x_max);
    try std.testing.expectEqual(@as(usize, 2), boxes[0].y_max);
    try std.testing.expectEqual(@as(usize, 2), boxes[1].x_min);
}
