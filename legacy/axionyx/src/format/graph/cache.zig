const std = @import("std");
const graph_types = @import("types.zig");

pub const AttrValue = graph_types.AttrValue;
pub const Graph = graph_types.Graph;
pub const ModuleNode = graph_types.ModuleNode;

pub fn cacheConvSpecs(model_graph: *const Graph, node: *ModuleNode) void {
    if (cachedConvSpecForModule(model_graph, node)) |cached| {
        node.cached_conv = cached;
    }
    for (node.children) |*child| {
        cacheConvSpecs(model_graph, child);
    }
}

pub fn cacheModuleAttrs(node: *ModuleNode) void {
    if (node.getAttr("c")) |value| {
        if (value.asInteger()) |integer| node.cached_attrs.c = @intCast(integer);
    }
    if (node.getAttr("add")) |value| {
        if (value.asBool()) |flag| node.cached_attrs.add = flag;
    }
    if (node.getAttr("nl")) |value| {
        if (value.asInteger()) |integer| node.cached_attrs.nl = @intCast(integer);
    }
    if (node.getAttr("nc")) |value| {
        if (value.asInteger()) |integer| node.cached_attrs.nc = @intCast(integer);
    }
    if (node.getAttr("reg_max")) |value| {
        if (value.asInteger()) |integer| node.cached_attrs.reg_max = @intCast(integer);
    }

    for (node.children) |*child| {
        cacheModuleAttrs(child);
    }
}

fn cachedConvSpecForModule(model_graph: *const Graph, module: *const ModuleNode) ?ModuleNode.CachedConvSpec {
    const is_wrapped_conv =
        std.mem.eql(u8, module.kind, "Conv") or
        std.mem.eql(u8, module.kind, "DWConv");
    const is_bare_conv2d = std.mem.eql(u8, module.kind, "Conv2d");
    if (!is_wrapped_conv and !is_bare_conv2d) return null;

    const stride = if (is_wrapped_conv)
        objectPair((module.getAttr("conv2d") orelse return null), "stride") orelse return null
    else
        nodePair(module, "stride") orelse return null;
    const padding = if (is_wrapped_conv)
        objectPair((module.getAttr("conv2d") orelse return null), "padding") orelse return null
    else
        nodePair(module, "padding") orelse return null;
    const groups = if (is_wrapped_conv)
        objectInteger((module.getAttr("conv2d") orelse return null), "groups") orelse return null
    else
        nodeInteger(module, "groups") orelse return null;

    var prefix_buffer: [256]u8 = undefined;
    const weight_prefix = weightPrefixForModulePath(&prefix_buffer, module.path) catch return null;

    var weight_name_buffer: [320]u8 = undefined;
    const weight_name = if (is_wrapped_conv)
        std.fmt.bufPrint(&weight_name_buffer, "{s}.conv.weight", .{weight_prefix}) catch return null
    else
        std.fmt.bufPrint(&weight_name_buffer, "{s}.weight", .{weight_prefix}) catch return null;

    var bias_name_buffer: [320]u8 = undefined;
    const bias_name = if (is_wrapped_conv)
        std.fmt.bufPrint(&bias_name_buffer, "{s}.conv.bias", .{weight_prefix}) catch return null
    else
        std.fmt.bufPrint(&bias_name_buffer, "{s}.bias", .{weight_prefix}) catch return null;

    const apply_silu = is_wrapped_conv and blk: {
        const activation = module.getAttr("activation") orelse break :blk false;
        break :blk std.mem.eql(u8, activation.asString() orelse "", "SiLU");
    };

    return .{
        .valid = true,
        .weight = model_graph.findTensor(weight_name) orelse return null,
        .bias = model_graph.findTensor(bias_name),
        .stride = stride,
        .padding = padding,
        .groups = @intCast(groups),
        .apply_silu = apply_silu,
    };
}

fn weightPrefixForModulePath(buffer: []u8, module_path: []const u8) ![]const u8 {
    if (std.mem.eql(u8, module_path, "model.model")) {
        return std.fmt.bufPrint(buffer, "model", .{});
    }

    if (std.mem.startsWith(u8, module_path, "model.model.")) {
        return std.fmt.bufPrint(buffer, "model.{s}", .{module_path["model.model.".len..]});
    }

    return std.fmt.bufPrint(buffer, "{s}", .{module_path});
}

fn nodePair(node: *const ModuleNode, key: []const u8) ?[2]usize {
    return pairFromValue(node.getAttr(key) orelse return null);
}

fn nodeInteger(node: *const ModuleNode, key: []const u8) ?i64 {
    return (node.getAttr(key) orelse return null).asInteger();
}

fn objectPair(object_value: *const AttrValue, key: []const u8) ?[2]usize {
    return pairFromValue(object_value.get(key) orelse return null);
}

fn objectInteger(object_value: *const AttrValue, key: []const u8) ?i64 {
    return (object_value.get(key) orelse return null).asInteger();
}

fn pairFromValue(value: *const AttrValue) ?[2]usize {
    if (value.asInteger()) |scalar| {
        const casted: usize = @intCast(scalar);
        return .{ casted, casted };
    }

    const items = value.asArray() orelse return null;
    if (items.len != 2) return null;
    return .{
        @intCast(items[0].asInteger() orelse return null),
        @intCast(items[1].asInteger() orelse return null),
    };
}
