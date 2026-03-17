const std = @import("std");

pub const AttrEntry = struct {
    key: []u8,
    value: AttrValue,

    pub fn deinit(self: *AttrEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        self.value.deinit(allocator);
        self.* = undefined;
    }
};

pub const AttrValue = union(enum) {
    null_value,
    bool_value: bool,
    integer_value: i64,
    float_value: f64,
    string_value: []u8,
    array_value: []AttrValue,
    object_value: []AttrEntry,

    pub fn deinit(self: *AttrValue, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .string_value => |value| allocator.free(value),
            .array_value => |items| {
                for (items) |*item| item.deinit(allocator);
                allocator.free(items);
            },
            .object_value => |entries| {
                for (entries) |*entry| entry.deinit(allocator);
                allocator.free(entries);
            },
            else => {},
        }
        self.* = undefined;
    }

    pub fn get(self: *const AttrValue, key: []const u8) ?*const AttrValue {
        return switch (self.*) {
            .object_value => |entries| blk: {
                for (entries) |*entry| {
                    if (std.mem.eql(u8, entry.key, key)) break :blk &entry.value;
                }
                break :blk null;
            },
            else => null,
        };
    }

    pub fn asBool(self: *const AttrValue) ?bool {
        return switch (self.*) {
            .bool_value => |value| value,
            else => null,
        };
    }

    pub fn asInteger(self: *const AttrValue) ?i64 {
        return switch (self.*) {
            .integer_value => |value| value,
            else => null,
        };
    }

    pub fn asFloat(self: *const AttrValue) ?f64 {
        return switch (self.*) {
            .float_value => |value| value,
            .integer_value => |value| @floatFromInt(value),
            else => null,
        };
    }

    pub fn asString(self: *const AttrValue) ?[]const u8 {
        return switch (self.*) {
            .string_value => |value| value,
            else => null,
        };
    }

    pub fn asArray(self: *const AttrValue) ?[]const AttrValue {
        return switch (self.*) {
            .array_value => |value| value,
            else => null,
        };
    }
};

pub const ModuleNode = struct {
    path: []u8,
    kind: []u8,
    attrs: []AttrEntry,
    children: []ModuleNode,

    pub fn deinit(self: *ModuleNode, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.kind);
        for (self.attrs) |*entry| entry.deinit(allocator);
        allocator.free(self.attrs);
        for (self.children) |*child| child.deinit(allocator);
        allocator.free(self.children);
        self.* = undefined;
    }

    pub fn findByPath(self: *const ModuleNode, target_path: []const u8) ?*const ModuleNode {
        if (std.mem.eql(u8, self.path, target_path)) return self;
        for (self.children) |*child| {
            if (child.findByPath(target_path)) |found| return found;
        }
        return null;
    }

    pub fn getAttr(self: *const ModuleNode, key: []const u8) ?*const AttrValue {
        for (self.attrs) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) return &entry.value;
        }
        return null;
    }
};

pub const TensorMeta = struct {
    name: []u8,
    rank: usize,
    shape: [4]usize,
    offset: usize,
    nbytes: usize,

    pub fn floatLen(self: *const TensorMeta) usize {
        return self.nbytes / @sizeOf(f32);
    }
};

pub const ExecutionNode = struct {
    index: usize,
    path: []u8,
    kind: []u8,
    from: []i64,
};

pub const Graph = struct {
    allocator: std.mem.Allocator,
    format_version: i64,
    model_name: []u8,
    class_count: i64,
    strides: []f32,
    tensors: []TensorMeta,
    execution_nodes: []ExecutionNode,
    module_tree: ModuleNode,

    pub fn deinit(self: *Graph) void {
        self.allocator.free(self.model_name);
        self.allocator.free(self.strides);
        for (self.tensors) |tensor| self.allocator.free(tensor.name);
        self.allocator.free(self.tensors);
        for (self.execution_nodes) |node| {
            self.allocator.free(node.path);
            self.allocator.free(node.kind);
            self.allocator.free(node.from);
        }
        self.allocator.free(self.execution_nodes);
        self.module_tree.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn findModule(self: *const Graph, target_path: []const u8) ?*const ModuleNode {
        return self.module_tree.findByPath(target_path);
    }

    pub fn findTensor(self: *const Graph, tensor_name: []const u8) ?*const TensorMeta {
        for (self.tensors) |*tensor| {
            if (std.mem.eql(u8, tensor.name, tensor_name)) return tensor;
        }
        return null;
    }
};

pub const Summary = struct {
    format_version: i64,
    model_name: []const u8,
    tensor_count: usize,
    execution_nodes: usize,
    class_count: i64,
};

pub fn load(allocator: std.mem.Allocator, graph_path: []const u8) !Graph {
    const cwd = std.fs.cwd();
    const contents = try cwd.readFileAlloc(allocator, graph_path, 64 * 1024 * 1024);
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

fn parseGraph(allocator: std.mem.Allocator, contents: []const u8) !Graph {
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

    return .{
        .allocator = allocator,
        .format_version = root.get("format_version").?.integer,
        .model_name = try allocator.dupe(u8, root.get("model_name").?.string),
        .class_count = metadata.get("class_count").?.integer,
        .strides = strides,
        .tensors = tensors,
        .execution_nodes = execution_nodes,
        .module_tree = module_tree,
    };
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

test "parseGraph exposes module tree and attrs" {
    const testing = std.testing;

    const raw =
        \\{
        \\  "format_version": 1,
        \\  "model_name": "mini",
        \\  "metadata": { "class_count": 2, "stride": [8.0, 16.0] },
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "model.0", "kind": "Conv", "from": [-1] }
        \\  ],
        \\  "module_tree": {
        \\    "path": "model",
        \\    "kind": "Root",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "model.0",
        \\        "kind": "Conv",
        \\        "attrs": {
        \\          "activation": "SiLU",
        \\          "conv2d": {
        \\            "out_channels": 16,
        \\            "kernel_size": [3, 3]
        \\          }
        \\        },
        \\        "children": []
        \\      }
        \\    ]
        \\  }
        \\}
    ;

    var model_graph = try parseGraph(testing.allocator, raw);
    defer model_graph.deinit();

    try testing.expectEqual(@as(usize, 2), model_graph.strides.len);
    try testing.expectApproxEqAbs(@as(f32, 8.0), model_graph.strides[0], 1e-6);

    const conv = model_graph.findModule("model.0").?;
    try testing.expectEqualStrings("Conv", conv.kind);
    try testing.expectEqualStrings("SiLU", conv.getAttr("activation").?.asString().?);

    const conv2d = conv.getAttr("conv2d").?;
    try testing.expectEqual(@as(i64, 16), conv2d.get("out_channels").?.asInteger().?);

    const kernel = conv2d.get("kernel_size").?.asArray().?;
    try testing.expectEqual(@as(i64, 3), kernel[0].asInteger().?);
    try testing.expectEqual(@as(i64, 3), kernel[1].asInteger().?);
}
