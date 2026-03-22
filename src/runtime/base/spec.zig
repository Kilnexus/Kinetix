const std = @import("std");
const graph = @import("graph");
const types = @import("types.zig");

pub const Activation = types.Activation;
pub const ConvSpec = types.ConvSpec;
pub const RuntimeError = types.RuntimeError;

pub fn resolveConvSpec(model_graph: *const graph.Graph, module_path: []const u8) RuntimeError!ConvSpec {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return resolveConvSpecNode(model_graph, module);
}

pub fn resolveConvSpecNode(model_graph: *const graph.Graph, module: *const graph.ModuleNode) RuntimeError!ConvSpec {
    if (module.cached_conv.valid) {
        return .{
            .weight = module.cached_conv.weight orelse return error.TensorNotFound,
            .bias = module.cached_conv.bias,
            .stride = module.cached_conv.stride,
            .padding = module.cached_conv.padding,
            .groups = module.cached_conv.groups,
            .activation = if (module.cached_conv.apply_silu) .silu else .identity,
        };
    }

    const module_path = module.path;

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

pub fn getNodePair(node: *const graph.ModuleNode, key: []const u8) RuntimeError![2]usize {
    const value = node.getAttr(key) orelse return error.MissingAttribute;
    return try pairFromValue(value);
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
