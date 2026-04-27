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
