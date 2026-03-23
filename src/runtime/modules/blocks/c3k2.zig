const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const bottleneck = @import("bottleneck.zig");
const c3k = @import("c3k.zig");
const conv = @import("conv.zig");
const types = @import("types.zig");
const utils = @import("../../base/utils.zig");

pub const Tensor = types.Tensor;
const c3k2_stack_part_limit = 8;

pub fn runC3k2Node(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
    module_runner: types.ModuleRunnerFn,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "C3k2")) return error.InvalidModuleKind;

    const chunk_channels = module.cached_attrs.c orelse @as(usize, @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));

    var stem = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer stem.deinit();

    if (stem.shape[1] != chunk_channels * 2) return ops.OpError.ShapeMismatch;

    const module_list = &module.children[2];
    if (module_list.children.len == 1) {
        const child = &module_list.children[0];
        const right_is_view = stem.shape[0] == 1;
        var right = if (right_is_view)
            try utils.sliceChannelsViewBatch1(&stem, chunk_channels, chunk_channels)
        else
            try utils.sliceChannels(allocator, &stem, chunk_channels, chunk_channels);
        defer if (!right_is_view) right.deinit();

        var child_out = if (std.mem.eql(u8, child.kind, "Bottleneck"))
            try bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, child, &right)
        else if (std.mem.eql(u8, child.kind, "C3k"))
            try c3k.runC3kNode(allocator, model_graph, weights_blob, child, &right, module_runner)
        else
            try module_runner(allocator, model_graph, weights_blob, child, &right);
        defer child_out.deinit();

        var concat = try Tensor.init(
            allocator,
            stem.shape[0],
            chunk_channels + right.shape[1] + child_out.shape[1],
            stem.shape[2],
            stem.shape[3],
        );
        defer concat.deinit();

        try ops.copyTensorBlock(&stem, &concat, 0);
        try ops.copyTensorBlock(&child_out, &concat, stem.shape[1]);
        return try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    }

    var parts_stack: [c3k2_stack_part_limit]Tensor = undefined;
    const parts_len = 2 + module_list.children.len;
    const use_stack_parts = parts_len <= parts_stack.len;
    var parts_heap: []Tensor = &.{};
    const parts = if (use_stack_parts)
        parts_stack[0..parts_len]
    else blk: {
        parts_heap = try allocator.alloc(Tensor, parts_len);
        break :blk parts_heap;
    };
    defer if (!use_stack_parts) allocator.free(parts_heap);

    var initialized_parts: usize = 0;
    const first_part_is_view = stem.shape[0] == 1;
    errdefer {
        const deinit_start: usize = if (first_part_is_view) 1 else 0;
        for (parts[deinit_start..initialized_parts]) |*part| part.deinit();
    }

    parts[0] = if (first_part_is_view)
        try utils.sliceChannelsViewBatch1(&stem, chunk_channels, chunk_channels)
    else
        try utils.sliceChannels(allocator, &stem, chunk_channels, chunk_channels);
    initialized_parts += 1;

    var current_index: usize = 0;
    for (module_list.children) |child| {
        parts[initialized_parts] = try module_runner(allocator, model_graph, weights_blob, &child, &parts[current_index]);
        current_index = initialized_parts;
        initialized_parts += 1;
    }
    defer {
        const deinit_start: usize = if (first_part_is_view) 1 else 0;
        for (parts[deinit_start..initialized_parts]) |*part| part.deinit();
    }

    var concat_channels: usize = chunk_channels;
    for (parts[0..initialized_parts]) |*part| concat_channels += part.shape[1];

    var concat = try Tensor.init(
        allocator,
        stem.shape[0],
        concat_channels,
        stem.shape[2],
        stem.shape[3],
    );
    defer concat.deinit();

    try ops.copyTensorBlock(&stem, &concat, 0);
    var channel_offset = stem.shape[1];
    for (parts[1..initialized_parts]) |*part| {
        try ops.copyTensorBlock(part, &concat, channel_offset);
        channel_offset += part.shape[1];
    }
    return try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
}

pub fn runC3k2ProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
    module_runner: types.ModuleRunnerFn,
) !types.ProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "C3k2")) return error.InvalidModuleKind;

    const chunk_channels = module.cached_attrs.c orelse @as(usize, @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));

    var profile = types.C3k2Profile{};
    var timer = try std.time.Timer.start();

    var stem = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer stem.deinit();

    if (stem.shape[1] != chunk_channels * 2) return ops.OpError.ShapeMismatch;

    const module_list = &module.children[2];
    if (module_list.children.len == 1) {
        const child = &module_list.children[0];
        profile.child_kind = child.kind;
        const right_is_view = stem.shape[0] == 1;
        var right = if (right_is_view)
            try utils.sliceChannelsViewBatch1(&stem, chunk_channels, chunk_channels)
        else
            try utils.sliceChannels(allocator, &stem, chunk_channels, chunk_channels);
        defer if (!right_is_view) right.deinit();

        timer.reset();
        var child_out = if (std.mem.eql(u8, child.kind, "Bottleneck")) blk: {
            const profiled = try bottleneck.runBottleneckProfileNode(allocator, model_graph, weights_blob, child, &right);
            profile.child_bottleneck = profiled.bottleneck_profile;
            break :blk profiled.output;
        } else if (std.mem.eql(u8, child.kind, "C3k")) blk: {
            const profiled = try c3k.runC3kProfileNode(allocator, model_graph, weights_blob, child, &right, module_runner);
            profile.child_c3k = profiled.c3k_profile;
            break :blk profiled.output;
        } else
            try module_runner(allocator, model_graph, weights_blob, child, &right);
        profile.child_ns = timer.read();
        defer child_out.deinit();

        timer.reset();
        var concat = try Tensor.init(
            allocator,
            stem.shape[0],
            chunk_channels + right.shape[1] + child_out.shape[1],
            stem.shape[2],
            stem.shape[3],
        );
        defer concat.deinit();

        timer.reset();
        try ops.copyTensorBlock(&stem, &concat, 0);
        try ops.copyTensorBlock(&child_out, &concat, stem.shape[1]);
        profile.concat_ns = timer.read();

        timer.reset();
        const output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
        profile.cv2_ns = timer.read();
        return .{ .output = output, .c3k2_profile = profile };
    }

    var parts_stack: [c3k2_stack_part_limit]Tensor = undefined;
    const parts_len = 2 + module_list.children.len;
    const use_stack_parts = parts_len <= parts_stack.len;
    var parts_heap: []Tensor = &.{};
    const parts = if (use_stack_parts)
        parts_stack[0..parts_len]
    else blk: {
        parts_heap = try allocator.alloc(Tensor, parts_len);
        break :blk parts_heap;
    };
    defer if (!use_stack_parts) allocator.free(parts_heap);

    var initialized_parts: usize = 0;
    const first_part_is_view = stem.shape[0] == 1;
    errdefer {
        const deinit_start: usize = if (first_part_is_view) 1 else 0;
        for (parts[deinit_start..initialized_parts]) |*part| part.deinit();
    }

    parts[0] = if (first_part_is_view)
        try utils.sliceChannelsViewBatch1(&stem, chunk_channels, chunk_channels)
    else
        try utils.sliceChannels(allocator, &stem, chunk_channels, chunk_channels);
    initialized_parts += 1;

    var current_index: usize = 0;
    profile.child_kind = "ModuleList";
    timer.reset();
    for (module_list.children) |child| {
        parts[initialized_parts] = try module_runner(allocator, model_graph, weights_blob, &child, &parts[current_index]);
        current_index = initialized_parts;
        initialized_parts += 1;
    }
    profile.child_ns = timer.read();
    defer {
        const deinit_start: usize = if (first_part_is_view) 1 else 0;
        for (parts[deinit_start..initialized_parts]) |*part| part.deinit();
    }

    var concat_channels: usize = chunk_channels;
    for (parts[0..initialized_parts]) |*part| concat_channels += part.shape[1];

    timer.reset();
    var concat = try Tensor.init(
        allocator,
        stem.shape[0],
        concat_channels,
        stem.shape[2],
        stem.shape[3],
    );
    defer concat.deinit();

    timer.reset();
    try ops.copyTensorBlock(&stem, &concat, 0);
    var channel_offset = stem.shape[1];
    for (parts[1..initialized_parts]) |*part| {
        try ops.copyTensorBlock(part, &concat, channel_offset);
        channel_offset += part.shape[1];
    }
    profile.concat_ns = timer.read();

    timer.reset();
    const output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    profile.cv2_ns = timer.read();
    return .{ .output = output, .c3k2_profile = profile };
}
