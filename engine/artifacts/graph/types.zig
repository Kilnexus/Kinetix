const std = @import("std");

pub const AttributeEntry = struct {
    key: []u8,
    value: AttributeValue,

    pub fn deinit(self: *AttributeEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        self.value.deinit(allocator);
        self.* = undefined;
    }
};

pub const AttributeValue = union(enum) {
    null_value,
    bool_value: bool,
    integer_value: i64,
    float_value: f64,
    string_value: []u8,
    array_value: []AttributeValue,
    object_value: []AttributeEntry,

    pub fn deinit(self: *AttributeValue, allocator: std.mem.Allocator) void {
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

    pub fn get(self: *const AttributeValue, key: []const u8) ?*const AttributeValue {
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

    pub fn asBool(self: *const AttributeValue) ?bool {
        return switch (self.*) {
            .bool_value => |value| value,
            else => null,
        };
    }

    pub fn asInteger(self: *const AttributeValue) ?i64 {
        return switch (self.*) {
            .integer_value => |value| value,
            else => null,
        };
    }

    pub fn asFloat(self: *const AttributeValue) ?f64 {
        return switch (self.*) {
            .float_value => |value| value,
            .integer_value => |value| @floatFromInt(value),
            else => null,
        };
    }

    pub fn asString(self: *const AttributeValue) ?[]const u8 {
        return switch (self.*) {
            .string_value => |value| value,
            else => null,
        };
    }

    pub fn asArray(self: *const AttributeValue) ?[]const AttributeValue {
        return switch (self.*) {
            .array_value => |value| value,
            else => null,
        };
    }
};

pub const ComponentNode = struct {
    path: []u8,
    kind: []u8,
    attrs: []AttributeEntry,
    children: []ComponentNode,

    pub fn deinit(self: *ComponentNode, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.kind);
        for (self.attrs) |*entry| entry.deinit(allocator);
        allocator.free(self.attrs);
        for (self.children) |*child| child.deinit(allocator);
        allocator.free(self.children);
        self.* = undefined;
    }

    pub fn findByPath(self: *const ComponentNode, target_path: []const u8) ?*const ComponentNode {
        if (std.mem.eql(u8, self.path, target_path)) return self;
        for (self.children) |*child| {
            if (child.findByPath(target_path)) |found| return found;
        }
        return null;
    }

    pub fn getAttr(self: *const ComponentNode, key: []const u8) ?*const AttributeValue {
        for (self.attrs) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) return &entry.value;
        }
        return null;
    }
};

pub const ArtifactTensor = struct {
    name: []u8,
    shape: []usize,
    offset: usize,
    nbytes: usize,

    pub fn deinit(self: *ArtifactTensor, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
        self.* = undefined;
    }

    pub fn rank(self: *const ArtifactTensor) usize {
        return self.shape.len;
    }
};

pub const ExecutionNode = struct {
    index: usize,
    path: []u8,
    kind: []u8,
    from: []i64,

    pub fn deinit(self: *ExecutionNode, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.kind);
        allocator.free(self.from);
        self.* = undefined;
    }
};

pub const PlanGraph = struct {
    allocator: std.mem.Allocator,
    format_version: i64,
    model_name: []u8,
    metadata: []AttributeEntry,
    tensors: []ArtifactTensor,
    execution_nodes: []ExecutionNode,
    execution_components: []?*const ComponentNode,
    execution_use_counts: []usize,
    component_tree: ComponentNode,
    component_index: std.StringHashMapUnmanaged(*const ComponentNode),
    tensor_index: std.StringHashMapUnmanaged(*const ArtifactTensor),

    pub fn deinit(self: *PlanGraph) void {
        self.component_index.deinit(self.allocator);
        self.tensor_index.deinit(self.allocator);
        self.allocator.free(self.model_name);
        for (self.metadata) |*entry| entry.deinit(self.allocator);
        self.allocator.free(self.metadata);
        for (self.tensors) |*tensor| tensor.deinit(self.allocator);
        self.allocator.free(self.tensors);
        for (self.execution_nodes) |*node| node.deinit(self.allocator);
        self.allocator.free(self.execution_nodes);
        self.allocator.free(self.execution_components);
        self.allocator.free(self.execution_use_counts);
        self.component_tree.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn getMetadata(self: *const PlanGraph, key: []const u8) ?*const AttributeValue {
        for (self.metadata) |*entry| {
            if (std.mem.eql(u8, entry.key, key)) return &entry.value;
        }
        return null;
    }

    pub fn findComponent(self: *const PlanGraph, target_path: []const u8) ?*const ComponentNode {
        return self.component_index.get(target_path);
    }

    pub fn findTensor(self: *const PlanGraph, tensor_name: []const u8) ?*const ArtifactTensor {
        return self.tensor_index.get(tensor_name);
    }
};

pub const Summary = struct {
    format_version: i64,
    model_name: []const u8,
    tensor_count: usize,
    execution_nodes: usize,
};
