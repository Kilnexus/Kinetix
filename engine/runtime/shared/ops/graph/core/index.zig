const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub const ElementwiseMode = enum { add, sub, mul, div };

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
