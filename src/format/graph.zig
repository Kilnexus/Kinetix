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
    pub const CachedConvSpec = struct {
        valid: bool = false,
        weight: ?*const TensorMeta = null,
        bias: ?*const TensorMeta = null,
        stride: [2]usize = .{ 1, 1 },
        padding: [2]usize = .{ 0, 0 },
        groups: usize = 1,
        apply_silu: bool = false,
    };

    pub const CachedAttrs = struct {
        c: ?usize = null,
        add: ?bool = null,
        nl: ?usize = null,
        nc: ?usize = null,
        reg_max: ?usize = null,
    };

    path: []u8,
    kind: []u8,
    attrs: []AttrEntry,
    children: []ModuleNode,
    cached_conv: CachedConvSpec,
    cached_attrs: CachedAttrs,

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
    execution_modules: []?*const ModuleNode,
    execution_use_counts: []usize,
    module_tree: ModuleNode,
    module_index: std.StringHashMapUnmanaged(*const ModuleNode),
    tensor_index: std.StringHashMapUnmanaged(*const TensorMeta),

    pub fn deinit(self: *Graph) void {
        self.module_index.deinit(self.allocator);
        self.tensor_index.deinit(self.allocator);
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
        self.allocator.free(self.execution_modules);
        self.allocator.free(self.execution_use_counts);
        self.module_tree.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn findModule(self: *const Graph, target_path: []const u8) ?*const ModuleNode {
        return self.module_index.get(target_path);
    }

    pub fn findTensor(self: *const Graph, tensor_name: []const u8) ?*const TensorMeta {
        return self.tensor_index.get(tensor_name);
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

    try indexGraph(&result);
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

fn indexGraph(model_graph: *Graph) !void {
    try model_graph.tensor_index.ensureTotalCapacity(model_graph.allocator, @intCast(model_graph.tensors.len));
    for (model_graph.tensors) |*tensor| {
        model_graph.tensor_index.putAssumeCapacity(tensor.name, tensor);
    }

    const module_count = countModules(&model_graph.module_tree);
    try model_graph.module_index.ensureTotalCapacity(model_graph.allocator, @intCast(module_count));
    indexModuleNode(&model_graph.module_index, &model_graph.module_tree);
    cacheConvSpecs(model_graph, &model_graph.module_tree);
    cacheModuleAttrs(&model_graph.module_tree);

    for (model_graph.execution_nodes, 0..) |*node, index| {
        model_graph.execution_modules[index] = executionModuleForPath(model_graph, node.path);
    }
    buildExecutionUseCounts(model_graph);
}

fn countModules(node: *const ModuleNode) usize {
    var total: usize = 1;
    for (node.children) |*child| total += countModules(child);
    return total;
}

fn indexModuleNode(index: *std.StringHashMapUnmanaged(*const ModuleNode), node: *const ModuleNode) void {
    index.putAssumeCapacity(node.path, node);
    for (node.children) |*child| indexModuleNode(index, child);
}

fn cacheConvSpecs(model_graph: *const Graph, node: *ModuleNode) void {
    if (cachedConvSpecForModule(model_graph, node)) |cached| {
        node.cached_conv = cached;
    }
    for (node.children) |*child| {
        cacheConvSpecs(model_graph, child);
    }
}

fn cacheModuleAttrs(node: *ModuleNode) void {
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

fn executionModuleForPath(model_graph: *const Graph, path: []const u8) ?*const ModuleNode {
    if (!std.mem.startsWith(u8, path, "model.")) return null;

    var buffer: [256]u8 = undefined;
    const module_path = std.fmt.bufPrint(&buffer, "model.model.{s}", .{path["model.".len..]}) catch return null;
    return model_graph.findModule(module_path);
}

fn buildExecutionUseCounts(model_graph: *Graph) void {
    @memset(model_graph.execution_use_counts, 0);
    for (model_graph.execution_nodes, 0..) |node, node_index| {
        for (node.from) |source| {
            if (source == -1) {
                if (node_index > 0) model_graph.execution_use_counts[node_index - 1] += 1;
            } else {
                model_graph.execution_use_counts[@intCast(source)] += 1;
            }
        }
    }
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
