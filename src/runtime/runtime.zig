const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const tensor_mod = @import("tensor");
const weights_mod = @import("weights");

pub const TensorDesc = struct {
    shape: [4]usize,
    len: usize,
};

pub const Tensor = tensor_mod.Tensor;

pub const Activation = enum {
    identity,
    silu,
};

pub const RuntimeError = error{
    BufferTooSmall,
    InvalidAttributeType,
    InvalidModuleKind,
    MissingAttribute,
    ModuleNotFound,
    TensorNotFound,
};

pub const ConvSpec = struct {
    weight: *const graph.TensorMeta,
    bias: ?*const graph.TensorMeta,
    stride: [2]usize,
    padding: [2]usize,
    groups: usize,
    activation: Activation,
};

pub fn shapeLen(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| total *= dim;
    return total;
}

pub fn resolveConvSpec(model_graph: *const graph.Graph, module_path: []const u8) RuntimeError!ConvSpec {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;

    const is_wrapped_conv =
        std.mem.eql(u8, module.kind, "Conv") or
        std.mem.eql(u8, module.kind, "DWConv");
    const is_bare_conv2d = std.mem.eql(u8, module.kind, "Conv2d");
    if (!is_wrapped_conv and !is_bare_conv2d) return error.InvalidModuleKind;

    const stride = if (is_wrapped_conv)
        try getObjectPair(module.getAttr("conv2d") orelse return error.MissingAttribute, "stride")
    else
        try getNodePair(module, "stride");

    const padding = if (is_wrapped_conv)
        try getObjectPair(module.getAttr("conv2d") orelse return error.MissingAttribute, "padding")
    else
        try getNodePair(module, "padding");

    const groups = if (is_wrapped_conv)
        try getObjectInteger(module.getAttr("conv2d") orelse return error.MissingAttribute, "groups")
    else
        try getNodeInteger(module, "groups");

    const activation = if (is_wrapped_conv)
        try parseActivation(module.getAttr("activation") orelse return error.MissingAttribute)
    else
        Activation.identity;

    var prefix_buffer: [256]u8 = undefined;
    const weight_prefix = try weightPrefixForModulePath(&prefix_buffer, module_path);

    var weight_name_buffer: [320]u8 = undefined;
    const weight_name = if (is_wrapped_conv)
        (std.fmt.bufPrint(&weight_name_buffer, "{s}.conv.weight", .{weight_prefix}) catch return error.BufferTooSmall)
    else
        (std.fmt.bufPrint(&weight_name_buffer, "{s}.weight", .{weight_prefix}) catch return error.BufferTooSmall);

    var bias_name_buffer: [320]u8 = undefined;
    const bias_name = if (is_wrapped_conv)
        (std.fmt.bufPrint(&bias_name_buffer, "{s}.conv.bias", .{weight_prefix}) catch return error.BufferTooSmall)
    else
        (std.fmt.bufPrint(&bias_name_buffer, "{s}.bias", .{weight_prefix}) catch return error.BufferTooSmall);

    return .{
        .weight = model_graph.findTensor(weight_name) orelse return error.TensorNotFound,
        .bias = model_graph.findTensor(bias_name),
        .stride = stride,
        .padding = padding,
        .groups = @intCast(groups),
        .activation = activation,
    };
}

pub fn runConvModule(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const spec = try resolveConvSpec(model_graph, module_path);
    const out_height = ((input.shape[2] + 2 * spec.padding[0] - spec.weight.shape[2]) / spec.stride[0]) + 1;
    const out_width = ((input.shape[3] + 2 * spec.padding[1] - spec.weight.shape[3]) / spec.stride[1]) + 1;

    var output = try Tensor.init(allocator, input.shape[0], spec.weight.shape[0], out_height, out_width);
    errdefer output.deinit();

    var weight_tensor = tensorView(spec.weight, weights_blob.slice(spec.weight));
    const bias_values = if (spec.bias) |bias_meta| weights_blob.slice(bias_meta) else null;

    try ops.conv2d(input, &weight_tensor, bias_values, &output, .{
        .stride_h = spec.stride[0],
        .stride_w = spec.stride[1],
        .pad_h = spec.padding[0],
        .pad_w = spec.padding[1],
        .groups = spec.groups,
    });
    applyActivation(&output, spec.activation);
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
    const cv1_path = try childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try childModulePath(&cv2_buffer, module_path, "cv2");

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
    const cv1_path = try childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try childModulePath(&cv2_buffer, module_path, "cv2");
    var pool_path_buffer: [256]u8 = undefined;
    const pool_path = try childModulePath(&pool_path_buffer, module_path, "m");

    const pool = model_graph.findModule(pool_path) orelse return error.ModuleNotFound;
    const stride = try getNodePair(pool, "stride");
    const padding = try getNodePair(pool, "padding");
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

pub fn weightPrefixForModulePath(buffer: []u8, module_path: []const u8) RuntimeError![]const u8 {
    if (std.mem.eql(u8, module_path, "model.model")) {
        return std.fmt.bufPrint(buffer, "model", .{}) catch return error.BufferTooSmall;
    }

    if (std.mem.startsWith(u8, module_path, "model.model.")) {
        return std.fmt.bufPrint(
            buffer,
            "model.{s}",
            .{module_path["model.model.".len..]},
        ) catch return error.BufferTooSmall;
    }

    return std.fmt.bufPrint(buffer, "{s}", .{module_path}) catch return error.BufferTooSmall;
}

pub fn printRoadmap(writer: anytype) !void {
    try writer.writeAll(
        \\Full runtime status:
        \\1. Graph and weights export: ready
        \\2. Zig graph loader: ready
        \\3. Primitive tensor ops: implemented
        \\4. Module-tree spec resolution: implemented
        \\5. Composite YOLO11s blocks: pending
        \\6. Detect + DFL + NMS: pending
        \\7. End-to-end parity check: pending
        \\
    );
}

fn parseActivation(value: *const graph.AttrValue) RuntimeError!Activation {
    const name = value.asString() orelse return error.InvalidAttributeType;
    if (std.mem.eql(u8, name, "Identity")) return .identity;
    if (std.mem.eql(u8, name, "SiLU")) return .silu;
    return error.InvalidAttributeType;
}

fn getNodeInteger(node: *const graph.ModuleNode, key: []const u8) RuntimeError!i64 {
    const value = node.getAttr(key) orelse return error.MissingAttribute;
    return value.asInteger() orelse return error.InvalidAttributeType;
}

fn getNodePair(node: *const graph.ModuleNode, key: []const u8) RuntimeError![2]usize {
    const value = node.getAttr(key) orelse return error.MissingAttribute;
    return try pairFromValue(value);
}

fn getObjectInteger(object_value: *const graph.AttrValue, key: []const u8) RuntimeError!i64 {
    const value = object_value.get(key) orelse return error.MissingAttribute;
    return value.asInteger() orelse return error.InvalidAttributeType;
}

fn getObjectPair(object_value: *const graph.AttrValue, key: []const u8) RuntimeError![2]usize {
    const value = object_value.get(key) orelse return error.MissingAttribute;
    return try pairFromValue(value);
}

fn pairFromValue(value: *const graph.AttrValue) RuntimeError![2]usize {
    if (value.asInteger()) |scalar| {
        const casted: usize = @intCast(scalar);
        return .{ casted, casted };
    }

    const items = value.asArray() orelse return error.InvalidAttributeType;
    if (items.len != 2) return error.InvalidAttributeType;

    return .{
        @intCast(items[0].asInteger() orelse return error.InvalidAttributeType),
        @intCast(items[1].asInteger() orelse return error.InvalidAttributeType),
    };
}

fn tensorView(meta: *const graph.TensorMeta, data: []const f32) Tensor {
    return .{
        .allocator = undefined,
        .data = @constCast(data),
        .shape = meta.shape,
    };
}

fn applyActivation(output: *Tensor, activation: Activation) void {
    switch (activation) {
        .identity => {},
        .silu => ops.siluInPlace(output),
    }
}

fn childModulePath(buffer: []u8, parent: []const u8, child: []const u8) RuntimeError![]const u8 {
    return std.fmt.bufPrint(buffer, "{s}.{s}", .{ parent, child }) catch return error.BufferTooSmall;
}

test "weightPrefixForModulePath normalizes exported module paths" {
    const testing = std.testing;

    var buffer: [128]u8 = undefined;
    try testing.expectEqualStrings(
        "model.2.cv1",
        try weightPrefixForModulePath(&buffer, "model.model.2.cv1"),
    );
    try testing.expectEqualStrings(
        "model.23.cv2.0.2",
        try weightPrefixForModulePath(&buffer, "model.model.23.cv2.0.2"),
    );
}

test "resolveConvSpec reads wrapped and bare conv metadata from exported graph" {
    const testing = std.testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();

    const wrapped = try resolveConvSpec(&model_graph, "model.model.2.cv1");
    try testing.expectEqualStrings("model.2.cv1.conv.weight", wrapped.weight.name);
    try testing.expectEqualStrings("model.2.cv1.conv.bias", wrapped.bias.?.name);
    try testing.expectEqual(Activation.silu, wrapped.activation);
    try testing.expectEqual(@as(usize, 1), wrapped.stride[0]);
    try testing.expectEqual(@as(usize, 0), wrapped.padding[0]);

    const bare = try resolveConvSpec(&model_graph, "model.model.23.cv2.0.2");
    try testing.expectEqualStrings("model.23.cv2.0.2.weight", bare.weight.name);
    try testing.expectEqualStrings("model.23.cv2.0.2.bias", bare.bias.?.name);
    try testing.expectEqual(Activation.identity, bare.activation);
    try testing.expectEqual(@as(usize, 1), bare.stride[0]);
    try testing.expectEqual(@as(usize, 0), bare.padding[0]);
}

test "runConvModule executes a wrapped conv from exported weights" {
    const testing = std.testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 64, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try runConvModule(testing.allocator, &model_graph, &weights_blob, "model.model.2.cv1", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 64, 8, 8 }, &output.shape);
}

test "runBottleneck preserves tensor shape for residual block" {
    const testing = std.testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 32, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try runBottleneck(testing.allocator, &model_graph, &weights_blob, "model.model.2.m.0", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &input.shape, &output.shape);
}

test "runSPPF executes pooled projection block" {
    const testing = std.testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 512, 4, 4);
    defer input.deinit();
    input.fill(0.0);

    var output = try runSPPF(testing.allocator, &model_graph, &weights_blob, "model.model.9", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 512, 4, 4 }, &output.shape);
}
