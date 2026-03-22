const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const psa = @import("psa.zig");
const spec = @import("../base/spec.zig");
const types = @import("../base/types.zig");
const utils = @import("../base/utils.zig");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;
const c3k2_stack_part_limit = 8;

pub const C3k2Profile = struct {
    cv1_ns: u64 = 0,
    child_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv2_ns: u64 = 0,
    child_kind: []const u8 = "",
    child_c3k: ?C3kProfile = null,
    child_bottleneck: ?BottleneckProfile = null,
};

pub const C3kProfile = struct {
    cv1_ns: u64 = 0,
    seq_ns: u64 = 0,
    cv2_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv3_ns: u64 = 0,
    seq_kind: []const u8 = "",
};

pub const BottleneckProfile = struct {
    cv1_ns: u64 = 0,
    cv2_ns: u64 = 0,
    add_ns: u64 = 0,
    has_add: bool = false,
};

pub const SPPFProfile = struct {
    cv1_ns: u64 = 0,
    pool1_ns: u64 = 0,
    pool2_ns: u64 = 0,
    pool3_ns: u64 = 0,
    concat_ns: u64 = 0,
    cv2_ns: u64 = 0,
};

pub const ProfiledTensor = struct {
    output: Tensor,
    c3k2_profile: C3k2Profile,
};

pub fn runModuleNodeDirect(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor {
    return runModuleNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runSPPFNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    return runSPPFInternal(allocator, model_graph, weights_blob, module, input);
}

pub fn runSPPFProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !SPPFProfiledTensor {
    return runSPPFProfileInternal(allocator, model_graph, weights_blob, module, input);
}

pub fn runC3k2ProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !ProfiledTensor {
    return runC3k2ProfileInternal(allocator, model_graph, weights_blob, module, input);
}

pub fn runConvModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runConvNode(allocator, model_graph, weights_blob, module, input);
}

fn runConvNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    const conv_spec = try spec.resolveConvSpecNode(model_graph, module);
    const out_height = ((input.shape[2] + 2 * conv_spec.padding[0] - conv_spec.weight.shape[2]) / conv_spec.stride[0]) + 1;
    const out_width = ((input.shape[3] + 2 * conv_spec.padding[1] - conv_spec.weight.shape[3]) / conv_spec.stride[1]) + 1;

    var output = try Tensor.init(allocator, input.shape[0], conv_spec.weight.shape[0], out_height, out_width);
    errdefer output.deinit();

    var weight_tensor = utils.tensorView(conv_spec.weight, weights_blob.slice(conv_spec.weight));
    const bias_values = if (conv_spec.bias) |bias_meta| weights_blob.slice(bias_meta) else null;

    try ops.conv2d(input, &weight_tensor, bias_values, &output, .{
        .stride_h = conv_spec.stride[0],
        .stride_w = conv_spec.stride[1],
        .pad_h = conv_spec.padding[0],
        .pad_w = conv_spec.padding[1],
        .groups = conv_spec.groups,
        .apply_silu = conv_spec.activation == .silu,
    });
    if (conv_spec.activation != .silu) {
        utils.applyActivation(&output, conv_spec.activation);
    }
    return output;
}

pub fn runBottleneck(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runBottleneckNode(allocator, model_graph, weights_blob, module, input);
}

pub const BottleneckProfiledTensor = struct {
    output: Tensor,
    bottleneck_profile: BottleneckProfile,
};

pub const SPPFProfiledTensor = struct {
    output: Tensor,
    sppf_profile: SPPFProfile,
};

pub fn runBottleneckProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !BottleneckProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runBottleneckProfileInternal(allocator, model_graph, weights_blob, module, input);
}

fn runBottleneckProfileInternal(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !BottleneckProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "Bottleneck")) return error.InvalidModuleKind;

    var profile = BottleneckProfile{};
    var timer = try std.time.Timer.start();

    var hidden = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer hidden.deinit();

    timer.reset();
    var output = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &hidden);
    profile.cv2_ns = timer.read();

    const has_add = module.cached_attrs.add orelse
        ((module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType);
    profile.has_add = has_add;
    if (has_add) {
        timer.reset();
        ops.addInPlaceUnchecked(&output, input);
        profile.add_ns = timer.read();
    }
    return .{ .output = output, .bottleneck_profile = profile };
}

pub fn runSPPF(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runSPPFInternal(allocator, model_graph, weights_blob, module, input);
}

fn runSPPFInternal(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "SPPF")) return error.InvalidModuleKind;

    const pool = &module.children[2];
    const stride = try spec.getNodePair(pool, "stride");
    const padding = try spec.getNodePair(pool, "padding");
    const kernel = .{ padding[0] * 2 + 1, padding[1] * 2 + 1 };

    var base = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer base.deinit();

    var pool1 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool1.deinit();
    try ops.maxPool2d(&base, &pool1, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var pool2 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool2.deinit();
    try ops.maxPool2d(&pool1, &pool2, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var pool3 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool3.deinit();
    try ops.maxPool2d(&pool2, &pool3, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var concat = try Tensor.init(
        allocator,
        base.shape[0],
        base.shape[1] * 4,
        base.shape[2],
        base.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &base, &pool1, &pool2, &pool3 };
    try ops.concatChannels(&inputs, &concat);

    return try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
}

pub fn runSPPFProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !SPPFProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runSPPFProfileInternal(allocator, model_graph, weights_blob, module, input);
}

fn runSPPFProfileInternal(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !SPPFProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "SPPF")) return error.InvalidModuleKind;

    const pool = &module.children[2];
    const stride = try spec.getNodePair(pool, "stride");
    const padding = try spec.getNodePair(pool, "padding");
    const kernel = .{ padding[0] * 2 + 1, padding[1] * 2 + 1 };

    var profile = SPPFProfile{};
    var timer = try std.time.Timer.start();

    var base = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer base.deinit();

    timer.reset();
    var pool1 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool1.deinit();
    try ops.maxPool2d(&base, &pool1, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool1_ns = timer.read();

    timer.reset();
    var pool2 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool2.deinit();
    try ops.maxPool2d(&pool1, &pool2, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool2_ns = timer.read();

    timer.reset();
    var pool3 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool3.deinit();
    try ops.maxPool2d(&pool2, &pool3, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool3_ns = timer.read();

    timer.reset();
    var concat = try Tensor.init(
        allocator,
        base.shape[0],
        base.shape[1] * 4,
        base.shape[2],
        base.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &base, &pool1, &pool2, &pool3 };
    try ops.concatChannels(&inputs, &concat);
    profile.concat_ns = timer.read();

    timer.reset();
    const output = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    profile.cv2_ns = timer.read();
    return .{ .output = output, .sppf_profile = profile };
}

pub fn runC3k(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runC3kNode(allocator, model_graph, weights_blob, module, input);
}

fn runBottleneckNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "Bottleneck")) return error.InvalidModuleKind;

    var hidden = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer hidden.deinit();

    var output = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &hidden);
    const has_add = module.cached_attrs.add orelse
        ((module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType);
    if (has_add) {
        ops.addInPlaceUnchecked(&output, input);
    }
    return output;
}

fn runC3kNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "C3k")) return error.InvalidModuleKind;

    var left = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer left.deinit();

    const seq_node = &module.children[3];
    if (seq_node.children.len == 2 and
        std.mem.eql(u8, seq_node.children[0].kind, "Bottleneck") and
        std.mem.eql(u8, seq_node.children[1].kind, "Bottleneck"))
    {
        const next0 = try runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[0], &left);
        left.deinit();

        left = next0;

        const next1 = try runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[1], &left);
        left.deinit();
        left = next1;
    } else {
        for (seq_node.children) |child| {
            const next = try runModuleNode(allocator, model_graph, weights_blob, &child, &left);
            left.deinit();
            left = next;
        }
    }

    var right = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], input);
    defer right.deinit();

    var concat = try Tensor.init(
        allocator,
        left.shape[0],
        left.shape[1] + right.shape[1],
        left.shape[2],
        left.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &left, &right };
    try ops.concatChannels(&inputs, &concat);

    return try runConvNode(allocator, model_graph, weights_blob, &module.children[2], &concat);
}

pub const C3kProfiledTensor = struct {
    output: Tensor,
    c3k_profile: C3kProfile,
};

pub fn runC3kProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !C3kProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runC3kProfileInternal(allocator, model_graph, weights_blob, module, input);
}

fn runC3kProfileInternal(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !C3kProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "C3k")) return error.InvalidModuleKind;

    var profile = C3kProfile{};
    var timer = try std.time.Timer.start();

    var left = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer left.deinit();

    const seq_node = &module.children[3];
    if (seq_node.children.len == 2 and
        std.mem.eql(u8, seq_node.children[0].kind, "Bottleneck") and
        std.mem.eql(u8, seq_node.children[1].kind, "Bottleneck"))
    {
        profile.seq_kind = "Bottleneckx2";
        timer.reset();
        const next0 = try runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[0], &left);
        left.deinit();
        left = next0;

        const next1 = try runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[1], &left);
        left.deinit();
        left = next1;
        profile.seq_ns = timer.read();
    } else {
        profile.seq_kind = seq_node.kind;
        timer.reset();
        for (seq_node.children) |child| {
            const next = try runModuleNode(allocator, model_graph, weights_blob, &child, &left);
            left.deinit();
            left = next;
        }
        profile.seq_ns = timer.read();
    }

    timer.reset();
    var right = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], input);
    profile.cv2_ns = timer.read();
    defer right.deinit();

    timer.reset();
    var concat = try Tensor.init(
        allocator,
        left.shape[0],
        left.shape[1] + right.shape[1],
        left.shape[2],
        left.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &left, &right };
    try ops.concatChannels(&inputs, &concat);
    profile.concat_ns = timer.read();

    timer.reset();
    const output = try runConvNode(allocator, model_graph, weights_blob, &module.children[2], &concat);
    profile.cv3_ns = timer.read();
    return .{ .output = output, .c3k_profile = profile };
}

pub fn runC3k2(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runC3k2Node(allocator, model_graph, weights_blob, module, input);
}

fn runC3k2Node(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "C3k2")) return error.InvalidModuleKind;

    const chunk_channels = module.cached_attrs.c orelse @as(usize, @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));

    var stem = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
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
            try runBottleneckNode(allocator, model_graph, weights_blob, child, &right)
        else if (std.mem.eql(u8, child.kind, "C3k"))
            try runC3kNode(allocator, model_graph, weights_blob, child, &right)
        else
            try runModuleNode(allocator, model_graph, weights_blob, child, &right);
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
        return try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
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
        parts[initialized_parts] = try runModuleNode(allocator, model_graph, weights_blob, &child, &parts[current_index]);
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
    return try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
}

pub fn runC3k2Profile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !ProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runC3k2ProfileInternal(allocator, model_graph, weights_blob, module, input);
}

fn runC3k2ProfileInternal(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !ProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "C3k2")) return error.InvalidModuleKind;

    const chunk_channels = module.cached_attrs.c orelse @as(usize, @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    ));

    var profile = C3k2Profile{};
    var timer = try std.time.Timer.start();

    var stem = try runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
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
            const profiled = try runBottleneckProfileInternal(allocator, model_graph, weights_blob, child, &right);
            profile.child_bottleneck = profiled.bottleneck_profile;
            break :blk profiled.output;
        }
        else if (std.mem.eql(u8, child.kind, "C3k")) blk: {
            const profiled = try runC3kProfileInternal(allocator, model_graph, weights_blob, child, &right);
            profile.child_c3k = profiled.c3k_profile;
            break :blk profiled.output;
        } else
            try runModuleNode(allocator, model_graph, weights_blob, child, &right);
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
        const output = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
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
        parts[initialized_parts] = try runModuleNode(allocator, model_graph, weights_blob, &child, &parts[current_index]);
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
    timer.reset();
    try ops.copyTensorBlock(&stem, &concat, 0);
    var channel_offset = stem.shape[1];
    for (parts[1..initialized_parts]) |*part| {
        try ops.copyTensorBlock(part, &concat, channel_offset);
        channel_offset += part.shape[1];
    }
    profile.concat_ns = timer.read();

    timer.reset();
    const output = try runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    profile.cv2_ns = timer.read();
    return .{ .output = output, .c3k2_profile = profile };
}

pub fn runModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) anyerror!Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runModuleNode(allocator, model_graph, weights_blob, module, input);
}

fn runModuleNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) anyerror!Tensor {
    if (std.mem.eql(u8, module.kind, "Identity")) {
        return input.clone();
    }
    if (std.mem.eql(u8, module.kind, "Sequential")) {
        if (module.children.len == 0) return input.clone();

        var current = try runModuleNode(allocator, model_graph, weights_blob, &module.children[0], input);
        errdefer current.deinit();
        for (module.children[1..]) |child| {
            const next = try runModuleNode(allocator, model_graph, weights_blob, &child, &current);
            current.deinit();
            current = next;
        }
        return current;
    }
    if (std.mem.eql(u8, module.kind, "Conv") or std.mem.eql(u8, module.kind, "DWConv") or std.mem.eql(u8, module.kind, "Conv2d")) {
        return runConvNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "Bottleneck")) {
        return runBottleneckNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "SPPF")) {
        return runSPPFInternal(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "C3k")) {
        return runC3kNode(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "C3k2")) {
        return runC3k2Node(allocator, model_graph, weights_blob, module, input);
    }
    if (std.mem.eql(u8, module.kind, "Attention")) {
        return psa.runAttention(allocator, model_graph, weights_blob, module.path, input);
    }
    if (std.mem.eql(u8, module.kind, "PSABlock")) {
        return psa.runPSABlock(allocator, model_graph, weights_blob, module.path, input);
    }
    if (std.mem.eql(u8, module.kind, "C2PSA")) {
        return psa.runC2PSA(allocator, model_graph, weights_blob, module.path, input);
    }
    return error.InvalidModuleKind;
}
