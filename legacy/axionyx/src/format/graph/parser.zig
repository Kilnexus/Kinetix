const std = @import("std");
const graph_index = @import("index.zig");
const graph_types = @import("types.zig");

pub const AttrEntry = graph_types.AttrEntry;
pub const AttrValue = graph_types.AttrValue;
pub const ExecutionNode = graph_types.ExecutionNode;
pub const Graph = graph_types.Graph;
pub const ModuleNode = graph_types.ModuleNode;
pub const Summary = graph_types.Summary;
pub const TensorMeta = graph_types.TensorMeta;

pub fn load(allocator: std.mem.Allocator, graph_path: []const u8) !Graph {
    const file = if (std.fs.path.isAbsolute(graph_path))
        try std.fs.openFileAbsolute(graph_path, .{})
    else
        try std.fs.cwd().openFile(graph_path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
    defer allocator.free(contents);

    return try parseGraph(allocator, contents);
}

pub fn loadSummary(allocator: std.mem.Allocator, graph_path: []const u8) !Summary {
    var model_graph = try load(allocator, graph_path);
    defer model_graph.deinit();

    return .{
        .format_version = model_graph.format_version,
        .model_name = try allocator.dupe(u8, model_graph.model_name),
        .tensor_count = model_graph.tensors.len,
        .execution_nodes = model_graph.execution_nodes.len,
        .class_count = model_graph.class_count,
    };
}

pub fn parseGraph(allocator: std.mem.Allocator, contents: []const u8) !Graph {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const metadata = root.get("metadata").?.object;
    const strides_json = metadata.get("stride").?.array;
    const tensors_json = root.get("tensors").?.array;
    const execution_plan_json = root.get("execution_plan").?.array;
    const module_tree_json = root.get("module_tree").?;

    var strides = try allocator.alloc(f32, strides_json.items.len);
    errdefer allocator.free(strides);
    for (strides_json.items, 0..) |stride_value, index| {
        strides[index] = @floatCast(stride_value.float);
    }

    var tensors = try allocator.alloc(TensorMeta, tensors_json.items.len);
    errdefer allocator.free(tensors);
    for (tensors_json.items, 0..) |tensor_value, index| {
        const tensor_object = tensor_value.object;
        const shape_json = tensor_object.get("shape").?.array;
        var shape = [_]usize{ 1, 1, 1, 1 };
        for (shape_json.items, 0..) |dim_value, dim_index| {
            if (dim_index >= shape.len) break;
            shape[dim_index] = @intCast(dim_value.integer);
        }

        tensors[index] = .{
            .name = try allocator.dupe(u8, tensor_object.get("name").?.string),
            .rank = shape_json.items.len,
            .shape = shape,
            .offset = @intCast(tensor_object.get("offset").?.integer),
            .nbytes = @intCast(tensor_object.get("nbytes").?.integer),
        };
    }

    var execution_nodes = try allocator.alloc(ExecutionNode, execution_plan_json.items.len);
    var initialized_nodes: usize = 0;
    errdefer {
        for (execution_nodes[0..initialized_nodes]) |node| {
            allocator.free(node.path);
            allocator.free(node.kind);
            allocator.free(node.from);
        }
        allocator.free(execution_nodes);
    }
    for (execution_plan_json.items, 0..) |node_value, index| {
        const node_object = node_value.object;
        const from_json = node_object.get("from").?.array;
        var from = try allocator.alloc(i64, from_json.items.len);
        for (from_json.items, 0..) |from_value, from_index| {
            from[from_index] = from_value.integer;
        }

        execution_nodes[index] = .{
            .index = @intCast(node_object.get("index").?.integer),
            .path = try allocator.dupe(u8, node_object.get("path").?.string),
            .kind = try allocator.dupe(u8, node_object.get("kind").?.string),
            .from = from,
        };
        initialized_nodes += 1;
    }

    var module_tree = try parseModuleNode(allocator, module_tree_json);
    errdefer module_tree.deinit(allocator);

    var result = Graph{
        .allocator = allocator,
        .format_version = root.get("format_version").?.integer,
        .model_name = try allocator.dupe(u8, root.get("model_name").?.string),
        .class_count = metadata.get("class_count").?.integer,
        .strides = strides,
        .tensors = tensors,
        .execution_nodes = execution_nodes,
        .execution_modules = try allocator.alloc(?*const ModuleNode, execution_nodes.len),
        .execution_use_counts = try allocator.alloc(usize, execution_nodes.len),
        .module_tree = module_tree,
        .module_index = .{},
        .tensor_index = .{},
    };
    errdefer result.deinit();

    try graph_index.indexGraph(&result);
    return result;
}

fn parseModuleNode(allocator: std.mem.Allocator, node_value: std.json.Value) !ModuleNode {
    const node_object = node_value.object;
    const attrs_object = node_object.get("attrs").?.object;
    const children_array = node_object.get("children").?.array;

    var attrs = try allocator.alloc(AttrEntry, attrs_object.count());
    var initialized_attrs: usize = 0;
    errdefer {
        for (attrs[0..initialized_attrs]) |*entry| entry.deinit(allocator);
        allocator.free(attrs);
    }

    var attr_iter = attrs_object.iterator();
    while (attr_iter.next()) |entry| {
        attrs[initialized_attrs] = .{
            .key = try allocator.dupe(u8, entry.key_ptr.*),
            .value = try parseAttrValue(allocator, entry.value_ptr.*),
        };
        initialized_attrs += 1;
    }

    var children = try allocator.alloc(ModuleNode, children_array.items.len);
    var initialized_children: usize = 0;
    errdefer {
        for (children[0..initialized_children]) |*child| child.deinit(allocator);
        allocator.free(children);
    }

    for (children_array.items) |child_value| {
        children[initialized_children] = try parseModuleNode(allocator, child_value);
        initialized_children += 1;
    }

    return .{
        .path = try allocator.dupe(u8, node_object.get("path").?.string),
        .kind = try allocator.dupe(u8, node_object.get("kind").?.string),
        .attrs = attrs,
        .children = children,
        .cached_conv = .{},
        .cached_attrs = .{},
    };
}

fn parseAttrValue(allocator: std.mem.Allocator, value: std.json.Value) !AttrValue {
    return switch (value) {
        .null => .null_value,
        .bool => |item| .{ .bool_value = item },
        .integer => |item| .{ .integer_value = item },
        .float => |item| .{ .float_value = item },
        .string => |item| .{ .string_value = try allocator.dupe(u8, item) },
        .array => |items| blk: {
            var values = try allocator.alloc(AttrValue, items.items.len);
            var initialized: usize = 0;
            errdefer {
                for (values[0..initialized]) |*owned| owned.deinit(allocator);
                allocator.free(values);
            }

            for (items.items) |child| {
                values[initialized] = try parseAttrValue(allocator, child);
                initialized += 1;
            }
            break :blk .{ .array_value = values };
        },
        .object => |object| blk: {
            var entries = try allocator.alloc(AttrEntry, object.count());
            var initialized: usize = 0;
            errdefer {
                for (entries[0..initialized]) |*entry| entry.deinit(allocator);
                allocator.free(entries);
            }

            var iter = object.iterator();
            while (iter.next()) |entry| {
                entries[initialized] = .{
                    .key = try allocator.dupe(u8, entry.key_ptr.*),
                    .value = try parseAttrValue(allocator, entry.value_ptr.*),
                };
                initialized += 1;
            }
            break :blk .{ .object_value = entries };
        },
        else => error.UnsupportedJsonValue,
    };
}
