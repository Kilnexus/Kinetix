const std = @import("std");

pub const Options = struct {
    threshold: f32 = 0.3,
    min_area: usize = 1,
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

    var stack = std.ArrayList(usize).init(allocator);
    defer stack.deinit();
    var boxes = std.ArrayList(Box).init(allocator);
    errdefer boxes.deinit();

    for (map, 0..) |value, index| {
        if (visited[index] or value < options.threshold) continue;
        const component = try floodFill(map, width, height, options.threshold, visited, &stack, index);
        if (component.area < options.min_area) continue;
        try boxes.append(.{
            .x_min = component.x_min,
            .y_min = component.y_min,
            .x_max = component.x_max,
            .y_max = component.y_max,
            .area = component.area,
            .score = component.score_sum / @as(f32, @floatFromInt(component.area)),
        });
    }

    return try boxes.toOwnedSlice();
}

fn floodFill(
    map: []const f32,
    width: usize,
    height: usize,
    threshold: f32,
    visited: []bool,
    stack: *std.ArrayList(usize),
    start: usize,
) !Component {
    stack.clearRetainingCapacity();
    try stack.append(start);
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
        const index = stack.pop().?;
        const x = index % width;
        const y = index / width;
        const score = map[index];

        component.area += 1;
        component.score_sum += score;
        component.x_min = @min(component.x_min, x);
        component.y_min = @min(component.y_min, y);
        component.x_max = @max(component.x_max, x);
        component.y_max = @max(component.y_max, y);

        try pushNeighbor(map, width, height, threshold, visited, stack, x, y, -1, 0);
        try pushNeighbor(map, width, height, threshold, visited, stack, x, y, 1, 0);
        try pushNeighbor(map, width, height, threshold, visited, stack, x, y, 0, -1);
        try pushNeighbor(map, width, height, threshold, visited, stack, x, y, 0, 1);
    }

    return component;
}

fn pushNeighbor(
    map: []const f32,
    width: usize,
    height: usize,
    threshold: f32,
    visited: []bool,
    stack: *std.ArrayList(usize),
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
    try stack.append(index);
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
        .{ .threshold = 0.5, .min_area = 2 },
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

