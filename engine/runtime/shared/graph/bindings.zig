const std = @import("std");

pub const DType = enum {
    i32,
    f32,

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .i32 => "i32",
            .f32 => "f32",
        };
    }
};

pub const Buffer = union(enum) {
    none,
    i32: []i32,
    f32: []f32,

    pub fn len(self: Buffer) usize {
        return switch (self) {
            .none => 0,
            .i32 => |items| items.len,
            .f32 => |items| items.len,
        };
    }

    fn deinit(self: Buffer, allocator: std.mem.Allocator) void {
        switch (self) {
            .none => {},
            .i32 => |items| allocator.free(items),
            .f32 => |items| allocator.free(items),
        }
    }
};

pub const TensorBinding = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    dtype: DType,
    shape: []usize,
    buffer: Buffer = .none,

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        dtype: DType,
        shape: []const usize,
        buffer: Buffer,
    ) !TensorBinding {
        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .dtype = dtype,
            .shape = try allocator.dupe(usize, shape),
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *TensorBinding) void {
        self.allocator.free(self.name);
        self.allocator.free(self.shape);
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn elementCount(self: TensorBinding) usize {
        if (self.shape.len == 0) return 0;
        var total: usize = 1;
        for (self.shape) |dim| total *= dim;
        return total;
    }

    pub fn hasConcreteBuffer(self: TensorBinding) bool {
        return self.buffer.len() == self.elementCount();
    }
};

pub const GraphInvocation = struct {
    allocator: std.mem.Allocator,
    stage: []const u8,
    model_file: []const u8,
    inputs: []TensorBinding,
    output_names: []const []const u8,

    pub fn init(
        allocator: std.mem.Allocator,
        stage: []const u8,
        model_file: []const u8,
        inputs: []TensorBinding,
        output_names: []const []const u8,
    ) !GraphInvocation {
        return .{
            .allocator = allocator,
            .stage = try allocator.dupe(u8, stage),
            .model_file = try allocator.dupe(u8, model_file),
            .inputs = inputs,
            .output_names = try copyStringList(allocator, output_names),
        };
    }

    pub fn deinit(self: *GraphInvocation) void {
        for (self.inputs) |*input| input.deinit();
        self.allocator.free(self.inputs);
        freeStringList(self.allocator, self.output_names);
        self.allocator.free(self.stage);
        self.allocator.free(self.model_file);
        self.* = undefined;
    }

    pub fn inputCount(self: GraphInvocation) usize {
        return self.inputs.len;
    }

    pub fn outputCount(self: GraphInvocation) usize {
        return self.output_names.len;
    }

    pub fn hasConcreteInputs(self: GraphInvocation) bool {
        for (self.inputs) |input| {
            if (!input.hasConcreteBuffer()) return false;
        }
        return true;
    }
};

pub fn copyStringList(allocator: std.mem.Allocator, items: []const []const u8) ![]const []const u8 {
    const out = try allocator.alloc([]const u8, items.len);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |item| allocator.free(item);
        allocator.free(out);
    }
    for (items, out) |item, *slot| {
        slot.* = try allocator.dupe(u8, item);
        filled += 1;
    }
    return out;
}

pub fn freeStringList(allocator: std.mem.Allocator, items: []const []const u8) void {
    for (items) |item| allocator.free(item);
    allocator.free(items);
}

test "graph invocation tracks concrete tensor bindings" {
    const allocator = std.testing.allocator;
    const values = try allocator.dupe(i32, &.{ 1, 2, 3, 4, 5, 6 });
    errdefer allocator.free(values);
    const inputs = try allocator.alloc(TensorBinding, 1);
    errdefer allocator.free(inputs);
    inputs[0] = try TensorBinding.init(allocator, "input_ids", .i32, &.{ 1, 2, 3 }, .{ .i32 = values });

    var invocation = try GraphInvocation.init(allocator, "prefill", "prefill.onnx", inputs, &.{ "global_hidden", "present_key_0" });
    defer invocation.deinit();

    try std.testing.expectEqual(@as(usize, 1), invocation.inputCount());
    try std.testing.expectEqual(@as(usize, 2), invocation.outputCount());
    try std.testing.expect(invocation.hasConcreteInputs());
    try std.testing.expectEqual(@as(usize, 6), invocation.inputs[0].elementCount());
}
