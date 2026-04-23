const std = @import("std");
const onnx_metadata = @import("../../onnx/metadata.zig");
const common = @import("common.zig");

const Tensor = common.Tensor;

pub fn sigmoid(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| {
        slot.* = 1.0 / (1.0 + @exp(-value));
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn leakyRelu(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const alpha = common.attributeFloat(node, "alpha") orelse 0.01;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| {
        slot.* = if (value >= 0) value else alpha * value;
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn gelu(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    const out = try allocator.alloc(f32, input.buffer.f32.len);
    errdefer allocator.free(out);
    for (input.buffer.f32, out) |value, *slot| {
        slot.* = geluValue(value);
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, input.shape),
        .buffer = .{ .f32 = out },
    };
}

pub fn swiglu(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 2) return error.InvalidOperatorArity;
    const gate = inputs[0].*;
    const up = inputs[1].*;
    if (gate.buffer != .f32 or up.buffer != .f32) return error.UnsupportedTensorDType;
    if (!gate.sameShape(up)) return error.ShapeMismatch;
    const out = try allocator.alloc(f32, gate.buffer.f32.len);
    errdefer allocator.free(out);
    for (gate.buffer.f32, up.buffer.f32, out) |gate_value, up_value, *slot| {
        slot.* = siluValue(gate_value) * up_value;
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, gate.shape),
        .buffer = .{ .f32 = out },
    };
}

fn geluValue(x: f32) f32 {
    const c = @sqrt(2.0 / std.math.pi);
    const inner = c * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

fn siluValue(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}
