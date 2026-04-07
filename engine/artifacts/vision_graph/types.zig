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
