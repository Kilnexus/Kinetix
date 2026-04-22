const std = @import("std");
const graph_index = @import("index.zig");
const graph_types = @import("types.zig");
const io = std.Options.debug_io;

pub const ArtifactTensor = graph_types.ArtifactTensor;
pub const AttributeEntry = graph_types.AttributeEntry;
pub const AttributeValue = graph_types.AttributeValue;
pub const ComponentNode = graph_types.ComponentNode;
pub const ExecutionNode = graph_types.ExecutionNode;
pub const PlanGraph = graph_types.PlanGraph;
pub const Summary = graph_types.Summary;

pub fn load(allocator: std.mem.Allocator, graph_path: []const u8) !PlanGraph {
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, graph_path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(contents);

    return try parseGraph(allocator, contents);
}

pub fn loadSummary(allocator: std.mem.Allocator, graph_path: []const u8) !Summary {
    var graph = try load(allocator, graph_path);
    defer graph.deinit();

    return .{
        .format_version = graph.format_version,
        .model_name = try allocator.dupe(u8, graph.model_name),
        .tensor_count = graph.tensors.len,
        .execution_nodes = graph.execution_nodes.len,
        .class_count = if (graph.getMetadata("class_count")) |value|
            if (value.asInteger()) |count|
                @intCast(count)
            else
                null
        else
            null,
    };
}

pub fn parseGraph(allocator: std.mem.Allocator, contents: []const u8) !PlanGraph {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    const metadata_value = root.get("metadata");
    const tensors_value = root.get("tensors");
    const execution_plan_value = root.get("execution_plan") orelse return error.MissingExecutionPlan;
    const component_tree_value = root.get("component_tree") orelse root.get("module_tree") orelse return error.MissingComponentTree;
    const model_name_value = root.get("model_name") orelse root.get("name") orelse return error.MissingModelName;
    if (model_name_value != .string) return error.InvalidModelName;

    const metadata = if (metadata_value) |value|
        try parseObjectEntries(allocator, value)
    else
        try allocator.alloc(AttributeEntry, 0);
    errdefer {
        for (metadata) |*entry| entry.deinit(allocator);
        allocator.free(metadata);
    }

    const tensors = if (tensors_value) |value|
        try parseTensors(allocator, value)
    else
        try allocator.alloc(ArtifactTensor, 0);
    errdefer {
        for (tensors) |*tensor| tensor.deinit(allocator);
        allocator.free(tensors);
    }

    const execution_nodes = try parseExecutionNodes(allocator, execution_plan_value);
    errdefer {
        for (execution_nodes) |*node| node.deinit(allocator);
        allocator.free(execution_nodes);
    }

    var component_tree = try parseComponentNode(allocator, component_tree_value);
    errdefer component_tree.deinit(allocator);

    var result = PlanGraph{
        .allocator = allocator,
        .format_version = if (root.get("format_version")) |value| switch (value) {
            .integer => |v| v,
            else => return error.InvalidFormatVersion,
        } else 1,
        .model_name = try allocator.dupe(u8, model_name_value.string),
        .metadata = metadata,
        .tensors = tensors,
        .execution_nodes = execution_nodes,
        .execution_components = try allocator.alloc(?*const ComponentNode, execution_nodes.len),
        .execution_use_counts = try allocator.alloc(usize, execution_nodes.len),
        .component_tree = component_tree,
        .component_index = .{},
        .tensor_index = .{},
    };
    errdefer result.deinit();

    try graph_index.indexGraph(&result);
    return result;
}

fn parseTensors(allocator: std.mem.Allocator, value: std.json.Value) ![]ArtifactTensor {
    if (value != .array) return error.InvalidTensorArray;
    const items = value.array.items;
    var tensors = try allocator.alloc(ArtifactTensor, items.len);
    var initialized: usize = 0;
    errdefer {
        for (tensors[0..initialized]) |*tensor| tensor.deinit(allocator);
        allocator.free(tensors);
    }

    for (items) |tensor_value| {
        if (tensor_value != .object) return error.InvalidTensorEntry;
        const object = tensor_value.object;
        const name_value = object.get("name") orelse return error.MissingTensorName;
        const shape_value = object.get("shape") orelse return error.MissingTensorShape;
        const offset_value = object.get("offset") orelse return error.MissingTensorOffset;
        const nbytes_value = object.get("nbytes") orelse return error.MissingTensorBytes;

        if (name_value != .string) return error.InvalidTensorName;
        const shape = try parseShape(allocator, shape_value);
        errdefer allocator.free(shape);

        tensors[initialized] = .{
            .name = try allocator.dupe(u8, name_value.string),
            .shape = shape,
            .offset = try jsonToUsize(offset_value),
            .nbytes = try jsonToUsize(nbytes_value),
        };
        initialized += 1;
    }

    return tensors;
}

fn parseExecutionNodes(allocator: std.mem.Allocator, value: std.json.Value) ![]ExecutionNode {
    if (value != .array) return error.InvalidExecutionPlan;
    const items = value.array.items;
    var nodes = try allocator.alloc(ExecutionNode, items.len);
    var initialized: usize = 0;
    errdefer {
        for (nodes[0..initialized]) |*node| node.deinit(allocator);
        allocator.free(nodes);
    }

    for (items) |node_value| {
        if (node_value != .object) return error.InvalidExecutionNode;
        const object = node_value.object;
        const path_value = object.get("path") orelse return error.MissingExecutionPath;
        const kind_value = object.get("kind") orelse return error.MissingExecutionKind;
        const from_value = object.get("from") orelse return error.MissingExecutionInputs;

        if (path_value != .string or kind_value != .string) return error.InvalidExecutionNode;

        const from = try parseInputs(allocator, from_value);
        errdefer allocator.free(from);

        nodes[initialized] = .{
            .index = if (object.get("index")) |index_value| try jsonToUsize(index_value) else initialized,
            .path = try allocator.dupe(u8, path_value.string),
            .kind = try allocator.dupe(u8, kind_value.string),
            .from = from,
        };
        initialized += 1;
    }

    return nodes;
}

fn parseComponentNode(allocator: std.mem.Allocator, value: std.json.Value) anyerror!ComponentNode {
    if (value != .object) return error.InvalidComponentNode;
    const object = value.object;
    const path_value = object.get("path") orelse return error.MissingComponentPath;
    const kind_value = object.get("kind") orelse return error.MissingComponentKind;
    const attrs_value = object.get("attrs");
    const children_value = object.get("children");

    if (path_value != .string or kind_value != .string) return error.InvalidComponentNode;

    const attrs = if (attrs_value) |item|
        try parseObjectEntries(allocator, item)
    else
        try allocator.alloc(AttributeEntry, 0);
    errdefer {
        for (attrs) |*entry| entry.deinit(allocator);
        allocator.free(attrs);
    }

    const children = if (children_value) |item|
        try parseChildren(allocator, item)
    else
        try allocator.alloc(ComponentNode, 0);
    errdefer {
        for (children) |*child| child.deinit(allocator);
        allocator.free(children);
    }

    return .{
        .path = try allocator.dupe(u8, path_value.string),
        .kind = try allocator.dupe(u8, kind_value.string),
        .attrs = attrs,
        .children = children,
    };
}

fn parseChildren(allocator: std.mem.Allocator, value: std.json.Value) anyerror![]ComponentNode {
    if (value != .array) return error.InvalidChildrenArray;
    const items = value.array.items;
    var children = try allocator.alloc(ComponentNode, items.len);
    var initialized: usize = 0;
    errdefer {
        for (children[0..initialized]) |*child| child.deinit(allocator);
        allocator.free(children);
    }

    for (items) |child_value| {
        children[initialized] = try parseComponentNode(allocator, child_value);
        initialized += 1;
    }

    return children;
}

fn parseObjectEntries(allocator: std.mem.Allocator, value: std.json.Value) anyerror![]AttributeEntry {
    if (value != .object) return error.InvalidObjectValue;
    var entries = try allocator.alloc(AttributeEntry, value.object.count());
    var initialized: usize = 0;
    errdefer {
        for (entries[0..initialized]) |*entry| entry.deinit(allocator);
        allocator.free(entries);
    }

    var iter = value.object.iterator();
    while (iter.next()) |entry| {
        entries[initialized] = .{
            .key = try allocator.dupe(u8, entry.key_ptr.*),
            .value = try parseAttributeValue(allocator, entry.value_ptr.*),
        };
        initialized += 1;
    }

    return entries;
}

fn parseAttributeValue(allocator: std.mem.Allocator, value: std.json.Value) anyerror!AttributeValue {
    return switch (value) {
        .null => .null_value,
        .bool => |item| .{ .bool_value = item },
        .integer => |item| .{ .integer_value = item },
        .float => |item| .{ .float_value = item },
        .string => |item| .{ .string_value = try allocator.dupe(u8, item) },
        .array => |items| blk: {
            var values = try allocator.alloc(AttributeValue, items.items.len);
            var initialized: usize = 0;
            errdefer {
                for (values[0..initialized]) |*owned| owned.deinit(allocator);
                allocator.free(values);
            }

            for (items.items) |child| {
                values[initialized] = try parseAttributeValue(allocator, child);
                initialized += 1;
            }
            break :blk .{ .array_value = values };
        },
        .object => |object| blk: {
            var entries = try allocator.alloc(AttributeEntry, object.count());
            var initialized: usize = 0;
            errdefer {
                for (entries[0..initialized]) |*entry| entry.deinit(allocator);
                allocator.free(entries);
            }

            var iter = object.iterator();
            while (iter.next()) |entry| {
                entries[initialized] = .{
                    .key = try allocator.dupe(u8, entry.key_ptr.*),
                    .value = try parseAttributeValue(allocator, entry.value_ptr.*),
                };
                initialized += 1;
            }
            break :blk .{ .object_value = entries };
        },
        else => error.UnsupportedJsonValue,
    };
}

fn parseShape(allocator: std.mem.Allocator, value: std.json.Value) ![]usize {
    if (value != .array) return error.InvalidShape;
    const items = value.array.items;
    var shape = try allocator.alloc(usize, items.len);
    for (items, 0..) |dim, index| {
        shape[index] = try jsonToUsize(dim);
    }
    return shape;
}

fn parseInputs(allocator: std.mem.Allocator, value: std.json.Value) ![]i64 {
    if (value != .array) return error.InvalidExecutionInputs;
    const items = value.array.items;
    var from = try allocator.alloc(i64, items.len);
    for (items, 0..) |item, index| {
        from[index] = switch (item) {
            .integer => |v| v,
            else => return error.InvalidExecutionInputs,
        };
    }
    return from;
}

fn jsonToUsize(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |v| @intCast(v),
        else => error.InvalidUnsignedInteger,
    };
}
