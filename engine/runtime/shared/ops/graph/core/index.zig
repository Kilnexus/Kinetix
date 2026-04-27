const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub const ElementwiseMode = enum { add, sub, mul, div };
pub const CompareMode = enum { equal, greater, less };
pub const LogicalMode = enum { and_op, or_op };
pub const UnaryFloatMode = enum { floor, ceil_op };

pub fn constant(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 0) return error.InvalidOperatorArity;
    for (node.attributes) |attribute| {
        if (!std.mem.eql(u8, attribute.name, "value")) continue;
        const literal = attribute.tensor orelse return error.UnsupportedConstantAttribute;
        if (literal.elem_type.raw == 7) {
            const tensor_shape = try common.dimsToShape(allocator, literal.dims);
            defer allocator.free(tensor_shape);
            return try Tensor.fromI64(allocator, tensor_shape, literal.int64_values);
        }
        return error.UnsupportedConstantTensorType;
    }
    return error.MissingConstantValue;
}

pub fn identity(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    return try inputs[0].clone(allocator);
}

pub fn elementwise(allocator: std.mem.Allocator, inputs: []const *const Tensor, mode: ElementwiseMode) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const lhs = inputs[0].*;
    const rhs = inputs[1].*;
    if (!lhs.sameShape(rhs)) return error.ShapeMismatch;
    if (lhs.buffer != .f32 or rhs.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, lhs.buffer.f32.len);
    errdefer allocator.free(out);
    for (lhs.buffer.f32, rhs.buffer.f32, out) |a, b, *slot| {
        slot.* = switch (mode) {
            .add => a + b,
            .sub => a - b,
            .mul => a * b,
            .div => a / b,
        };
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, lhs.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn whereOp(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 3) return error.InvalidOperatorArity;
    const condition = inputs[0].*;
    const x = inputs[1].*;
    const y = inputs[2].*;
    if (!condition.sameShape(x) or !x.sameShape(y)) return error.ShapeMismatch;
    if (condition.buffer != .i64 or x.buffer != .f32 or y.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, x.buffer.f32.len);
    errdefer allocator.free(out);
    for (condition.buffer.i64, x.buffer.f32, y.buffer.f32, out) |cond, x_value, y_value, *slot| {
        slot.* = if (cond != 0) x_value else y_value;
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, x.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn compare(allocator: std.mem.Allocator, inputs: []const *const Tensor, mode: CompareMode) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const lhs = inputs[0].*;
    const rhs = inputs[1].*;
    if (!lhs.sameShape(rhs)) return error.ShapeMismatch;
    const out = try allocator.alloc(i64, lhs.elementCount());
    errdefer allocator.free(out);
    switch (lhs.buffer) {
        .f32 => |lhs_values| {
            if (rhs.buffer != .f32) return error.UnsupportedTensorDType;
            for (lhs_values, rhs.buffer.f32, out) |a, b, *slot| slot.* = @intFromBool(compareValues(a, b, mode));
        },
        .i32 => |lhs_values| {
            if (rhs.buffer != .i32) return error.UnsupportedTensorDType;
            for (lhs_values, rhs.buffer.i32, out) |a, b, *slot| slot.* = @intFromBool(compareValues(a, b, mode));
        },
        .i64 => |lhs_values| {
            if (rhs.buffer != .i64) return error.UnsupportedTensorDType;
            for (lhs_values, rhs.buffer.i64, out) |a, b, *slot| slot.* = @intFromBool(compareValues(a, b, mode));
        },
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, lhs.shape),
        .buffer = .{ .i64 = out },
    };
}

pub fn logical(allocator: std.mem.Allocator, inputs: []const *const Tensor, mode: LogicalMode) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const lhs = inputs[0].*;
    const rhs = inputs[1].*;
    if (!lhs.sameShape(rhs) or lhs.buffer != .i64 or rhs.buffer != .i64) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(i64, lhs.buffer.i64.len);
    errdefer allocator.free(out);
    for (lhs.buffer.i64, rhs.buffer.i64, out) |a, b, *slot| {
        slot.* = switch (mode) {
            .and_op => @intFromBool(a != 0 and b != 0),
            .or_op => @intFromBool(a != 0 or b != 0),
        };
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, lhs.shape),
        .buffer = .{ .i64 = out },
    };
}

pub fn notOp(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .i64) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(i64, input.buffer.i64.len);
    errdefer allocator.free(out);
    for (input.buffer.i64, out) |value, *slot| slot.* = @intFromBool(value == 0);
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .i64 = out },
    };
}

pub fn unaryFloat(allocator: std.mem.Allocator, inputs: []const *const Tensor, mode: UnaryFloatMode) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| {
        slot.* = switch (mode) {
            .floor => @floor(value),
            .ceil_op => @ceil(value),
        };
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn range(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 3) return error.InvalidOperatorArity;
    const start = inputs[0].*;
    const limit = inputs[1].*;
    const delta = inputs[2].*;
    if (start.elementCount() != 1 or limit.elementCount() != 1 or delta.elementCount() != 1) return error.ShapeMismatch;
    return switch (start.buffer) {
        .i64 => try rangeI64(allocator, start.buffer.i64[0], try scalarI64(limit), try scalarI64(delta)),
        .i32 => try rangeI64(allocator, start.buffer.i32[0], try scalarI64(limit), try scalarI64(delta)),
        .f32 => try rangeF32(allocator, start.buffer.f32[0], try scalarF32(limit), try scalarF32(delta)),
    };
}

fn compareValues(a: anytype, b: @TypeOf(a), mode: CompareMode) bool {
    return switch (mode) {
        .equal => a == b,
        .greater => a > b,
        .less => a < b,
    };
}

fn scalarI64(tensor: Tensor) !i64 {
    if (tensor.elementCount() != 1) return error.ShapeMismatch;
    return switch (tensor.buffer) {
        .i64 => |values| values[0],
        .i32 => |values| values[0],
        .f32 => |values| @intFromFloat(values[0]),
    };
}

fn rangeI64(allocator: std.mem.Allocator, start: i64, limit: i64, delta: i64) !Tensor {
    if (delta == 0) return error.InvalidOperatorAttribute;
    const len = rangeLength(i64, start, limit, delta);
    const out = try allocator.alloc(i64, len);
    errdefer allocator.free(out);
    var value = start;
    for (out) |*slot| {
        slot.* = value;
        value += delta;
    }
    return .{ .allocator = allocator, .shape = try allocator.dupe(usize, &.{len}), .buffer = .{ .i64 = out } };
}

fn rangeF32(allocator: std.mem.Allocator, start: f32, limit: f32, delta: f32) !Tensor {
    if (delta == 0) return error.InvalidOperatorAttribute;
    const len = rangeLength(f32, start, limit, delta);
    const out = try allocator.alloc(f32, len);
    errdefer allocator.free(out);
    var value = start;
    for (out) |*slot| {
        slot.* = value;
        value += delta;
    }
    return .{ .allocator = allocator, .shape = try allocator.dupe(usize, &.{len}), .buffer = .{ .f32 = out } };
}

fn rangeLength(comptime T: type, start: T, limit: T, delta: T) usize {
    var len: usize = 0;
    var value = start;
    while ((delta > 0 and value < limit) or (delta < 0 and value > limit)) : (value += delta) len += 1;
    return len;
}

pub fn relu(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| slot.* = @max(value, 0);
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn clip(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 1 or inputs.len > 3) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const min_value = if (inputs.len >= 2) try scalarF32(inputs[1].*) else common.attributeFloat(node, "min") orelse -std.math.inf(f32);
    const max_value = if (inputs.len >= 3) try scalarF32(inputs[2].*) else common.attributeFloat(node, "max") orelse std.math.inf(f32);
    if (min_value > max_value) return error.InvalidOperatorAttribute;

    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| slot.* = @min(@max(value, min_value), max_value);
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

fn scalarF32(tensor: Tensor) !f32 {
    if (tensor.elementCount() != 1) return error.ShapeMismatch;
    return switch (tensor.buffer) {
        .f32 => |values| values[0],
        .i32 => |values| @floatFromInt(values[0]),
        .i64 => |values| @floatFromInt(values[0]),
    };
}

pub fn cast(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const to = common.attributeInt(node, "to") orelse return error.MissingOperatorAttribute;
    const input = inputs[0].*;
    return switch (to) {
        1 => switch (input.buffer) {
            .f32 => |values| try Tensor.fromF32(allocator, input.shape, values),
            .i32 => |values| blk: {
                const out = try allocator.alloc(f32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @floatFromInt(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .f32 = out } };
            },
            .i64 => |values| blk: {
                const out = try allocator.alloc(f32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @floatFromInt(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .f32 = out } };
            },
        },
        6 => switch (input.buffer) {
            .i32 => |values| try Tensor.fromI32(allocator, input.shape, values),
            .i64 => |values| blk: {
                const out = try allocator.alloc(i32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intCast(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i32 = out } };
            },
            .f32 => |values| blk: {
                const out = try allocator.alloc(i32, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intFromFloat(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i32 = out } };
            },
        },
        7 => switch (input.buffer) {
            .i64 => |values| try Tensor.fromI64(allocator, input.shape, values),
            .i32 => |values| blk: {
                const out = try allocator.alloc(i64, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = value;
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i64 = out } };
            },
            .f32 => |values| blk: {
                const out = try allocator.alloc(i64, values.len);
                errdefer allocator.free(out);
                for (values, out) |value, *slot| slot.* = @intFromFloat(value);
                break :blk Tensor{ .allocator = allocator, .shape = try allocator.dupe(usize, input.shape), .buffer = .{ .i64 = out } };
            },
        },
        else => error.UnsupportedTensorDType,
    };
}
