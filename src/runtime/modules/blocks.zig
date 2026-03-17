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

pub fn runConvModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const conv_spec = try spec.resolveConvSpec(model_graph, module_path);
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
    });
    utils.applyActivation(&output, conv_spec.activation);
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
    if (!std.mem.eql(u8, module.kind, "Bottleneck")) return error.InvalidModuleKind;

    var cv1_buffer: [256]u8 = undefined;
    const cv1_path = try utils.childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try utils.childModulePath(&cv2_buffer, module_path, "cv2");

    var hidden = try runConvModule(allocator, model_graph, weights_blob, cv1_path, input);
    defer hidden.deinit();

    var output = try runConvModule(allocator, model_graph, weights_blob, cv2_path, &hidden);
    if ((module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType) {
        if (!output.sameShape(input)) return ops.OpError.ShapeMismatch;
        for (output.data, input.data) |*dst, src| dst.* += src;
    }
    return output;
}

pub fn runSPPF(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "SPPF")) return error.InvalidModuleKind;

    var cv1_buffer: [256]u8 = undefined;
    const cv1_path = try utils.childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try utils.childModulePath(&cv2_buffer, module_path, "cv2");
    var pool_path_buffer: [256]u8 = undefined;
    const pool_path = try utils.childModulePath(&pool_path_buffer, module_path, "m");

    const pool = model_graph.findModule(pool_path) orelse return error.ModuleNotFound;
    const stride = try spec.getNodePair(pool, "stride");
    const padding = try spec.getNodePair(pool, "padding");
    const kernel = .{ padding[0] * 2 + 1, padding[1] * 2 + 1 };

    var base = try runConvModule(allocator, model_graph, weights_blob, cv1_path, input);
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

    return try runConvModule(allocator, model_graph, weights_blob, cv2_path, &concat);
}

pub fn runC3k(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "C3k")) return error.InvalidModuleKind;

    var cv1_buffer: [256]u8 = undefined;
    const cv1_path = try utils.childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try utils.childModulePath(&cv2_buffer, module_path, "cv2");
    var cv3_buffer: [256]u8 = undefined;
    const cv3_path = try utils.childModulePath(&cv3_buffer, module_path, "cv3");
    var seq_buffer: [256]u8 = undefined;
    const seq_path = try utils.childModulePath(&seq_buffer, module_path, "m");

    var left = try runConvModule(allocator, model_graph, weights_blob, cv1_path, input);
    defer left.deinit();

    const seq_node = model_graph.findModule(seq_path) orelse return error.ModuleNotFound;
    for (seq_node.children) |child| {
        const next = try runModule(allocator, model_graph, weights_blob, child.path, &left);
        left.deinit();
        left = next;
    }

    var right = try runConvModule(allocator, model_graph, weights_blob, cv2_path, input);
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

    return try runConvModule(allocator, model_graph, weights_blob, cv3_path, &concat);
}

pub fn runC3k2(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "C3k2")) return error.InvalidModuleKind;

    const chunk_channels: usize = @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );

    var cv1_buffer: [256]u8 = undefined;
    const cv1_path = try utils.childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try utils.childModulePath(&cv2_buffer, module_path, "cv2");
    var list_buffer: [256]u8 = undefined;
    const list_path = try utils.childModulePath(&list_buffer, module_path, "m");

    var stem = try runConvModule(allocator, model_graph, weights_blob, cv1_path, input);
    defer stem.deinit();

    if (stem.shape[1] != chunk_channels * 2) return ops.OpError.ShapeMismatch;

    const module_list = model_graph.findModule(list_path) orelse return error.ModuleNotFound;
    var parts = try allocator.alloc(Tensor, 2 + module_list.children.len);
    defer allocator.free(parts);

    var initialized_parts: usize = 0;
    errdefer {
        for (parts[0..initialized_parts]) |*part| part.deinit();
    }

    parts[0] = try utils.sliceChannels(allocator, &stem, 0, chunk_channels);
    initialized_parts += 1;
    parts[1] = try utils.sliceChannels(allocator, &stem, chunk_channels, chunk_channels);
    initialized_parts += 1;

    var current_index: usize = 1;
    for (module_list.children) |child| {
        parts[initialized_parts] = try runModule(allocator, model_graph, weights_blob, child.path, &parts[current_index]);
        current_index = initialized_parts;
        initialized_parts += 1;
    }
    defer {
        for (parts[0..initialized_parts]) |*part| part.deinit();
    }

    var input_ptrs = try allocator.alloc(*const Tensor, initialized_parts);
    defer allocator.free(input_ptrs);

    var concat_channels: usize = 0;
    for (parts[0..initialized_parts], 0..) |*part, index| {
        input_ptrs[index] = part;
        concat_channels += part.shape[1];
    }

    var concat = try Tensor.init(
        allocator,
        stem.shape[0],
        concat_channels,
        stem.shape[2],
        stem.shape[3],
    );
    defer concat.deinit();
    try ops.concatChannels(input_ptrs, &concat);

    return try runConvModule(allocator, model_graph, weights_blob, cv2_path, &concat);
}

pub fn runModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) anyerror!Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;

    if (std.mem.eql(u8, module.kind, "Identity")) {
        return input.clone();
    }
    if (std.mem.eql(u8, module.kind, "Sequential")) {
        if (module.children.len == 0) return input.clone();

        var current = try runModule(allocator, model_graph, weights_blob, module.children[0].path, input);
        errdefer current.deinit();
        for (module.children[1..]) |child| {
            const next = try runModule(allocator, model_graph, weights_blob, child.path, &current);
            current.deinit();
            current = next;
        }
        return current;
    }
    if (std.mem.eql(u8, module.kind, "Conv") or std.mem.eql(u8, module.kind, "DWConv") or std.mem.eql(u8, module.kind, "Conv2d")) {
        return runConvModule(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "Bottleneck")) {
        return runBottleneck(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "SPPF")) {
        return runSPPF(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "C3k")) {
        return runC3k(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "C3k2")) {
        return runC3k2(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "Attention")) {
        return psa.runAttention(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "PSABlock")) {
        return psa.runPSABlock(allocator, model_graph, weights_blob, module_path, input);
    }
    if (std.mem.eql(u8, module.kind, "C2PSA")) {
        return psa.runC2PSA(allocator, model_graph, weights_blob, module_path, input);
    }
    return error.InvalidModuleKind;
}
