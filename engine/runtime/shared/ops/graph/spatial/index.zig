const std = @import("std");
const onnx_metadata = @import("shared_graph").onnx.metadata;
const common = @import("../common.zig");

const Tensor = common.Tensor;

pub fn conv(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 2 or inputs.len > 3) return error.InvalidOperatorArity;
    const x = inputs[0].*;
    const w = inputs[1].*;
    if (x.buffer != .f32 or w.buffer != .f32) return error.UnsupportedTensorDType;
    if (x.shape.len != 4 or w.shape.len != 4) return error.UnsupportedTensorRank;
    const n = x.shape[0];
    const in_channels = x.shape[1];
    const in_h = x.shape[2];
    const in_w = x.shape[3];
    const out_channels = w.shape[0];
    const kernel_channels = w.shape[1];
    const kernel_h = w.shape[2];
    const kernel_w = w.shape[3];
    const group: usize = @intCast(common.attributeInt(node, "group") orelse 1);
    if (group == 0 or in_channels % group != 0 or out_channels % group != 0) return error.InvalidGroups;
    if (kernel_channels != in_channels / group) return error.ShapeMismatch;
    const strides = try pairAttribute(allocator, node, "strides", 1);
    const dilations = try pairAttribute(allocator, node, "dilations", 1);
    const pads = try padsAttribute(allocator, node);
    if (dilations.h != 1 or dilations.w != 1) return error.UnsupportedOperatorAttribute;
    const out_h = try convOutputDim(in_h, kernel_h, pads.top, pads.bottom, strides.h);
    const out_w = try convOutputDim(in_w, kernel_w, pads.left, pads.right, strides.w);
    const out = try allocator.alloc(f32, n * out_channels * out_h * out_w);
    errdefer allocator.free(out);
    const bias = if (inputs.len == 3) inputs[2].* else null;
    if (bias) |bias_tensor| {
        if (bias_tensor.buffer != .f32 or bias_tensor.elementCount() != out_channels) return error.ShapeMismatch;
    }

    const out_channel_per_group = out_channels / group;
    const in_channel_per_group = in_channels / group;
    for (0..n) |batch| {
        for (0..out_channels) |oc| {
            const group_index = oc / out_channel_per_group;
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var sum: f32 = if (bias) |bias_tensor| bias_tensor.buffer.f32[oc] else 0;
                    for (0..in_channel_per_group) |ic_local| {
                        const ic = group_index * in_channel_per_group + ic_local;
                        for (0..kernel_h) |ky| {
                            const in_y = @as(isize, @intCast(oy * strides.h + ky)) - @as(isize, @intCast(pads.top));
                            if (in_y < 0 or in_y >= @as(isize, @intCast(in_h))) continue;
                            for (0..kernel_w) |kx| {
                                const in_x = @as(isize, @intCast(ox * strides.w + kx)) - @as(isize, @intCast(pads.left));
                                if (in_x < 0 or in_x >= @as(isize, @intCast(in_w))) continue;
                                const x_index = ((batch * in_channels + ic) * in_h + @as(usize, @intCast(in_y))) * in_w + @as(usize, @intCast(in_x));
                                const w_index = ((oc * kernel_channels + ic_local) * kernel_h + ky) * kernel_w + kx;
                                sum += x.buffer.f32[x_index] * w.buffer.f32[w_index];
                            }
                        }
                    }
                    out[((batch * out_channels + oc) * out_h + oy) * out_w + ox] = sum;
                }
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ n, out_channels, out_h, out_w }),
        .buffer = .{ .f32 = out },
    };
}

pub fn convTranspose(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len < 2 or inputs.len > 3) return error.InvalidOperatorArity;
    const x = inputs[0].*;
    const w = inputs[1].*;
    if (x.buffer != .f32 or w.buffer != .f32) return error.UnsupportedTensorDType;
    if (x.shape.len != 4 or w.shape.len != 4) return error.UnsupportedTensorRank;
    const n = x.shape[0];
    const in_channels = x.shape[1];
    const in_h = x.shape[2];
    const in_w = x.shape[3];
    const weight_in_channels = w.shape[0];
    const out_channels_per_group = w.shape[1];
    const kernel_h = w.shape[2];
    const kernel_w = w.shape[3];
    const group: usize = @intCast(common.attributeInt(node, "group") orelse 1);
    if (group == 0 or in_channels % group != 0 or weight_in_channels != in_channels) return error.InvalidGroups;
    const out_channels = out_channels_per_group * group;
    const strides = try pairAttribute(allocator, node, "strides", 1);
    const dilations = try pairAttribute(allocator, node, "dilations", 1);
    const pads = try padsAttribute(allocator, node);
    if (dilations.h != 1 or dilations.w != 1) return error.UnsupportedOperatorAttribute;
    const out_h = try convTransposeOutputDim(in_h, kernel_h, pads.top, pads.bottom, strides.h);
    const out_w = try convTransposeOutputDim(in_w, kernel_w, pads.left, pads.right, strides.w);
    const out = try allocator.alloc(f32, n * out_channels * out_h * out_w);
    errdefer allocator.free(out);
    @memset(out, 0);
    const bias = if (inputs.len == 3) inputs[2].* else null;
    if (bias) |bias_tensor| {
        if (bias_tensor.buffer != .f32 or bias_tensor.elementCount() != out_channels) return error.ShapeMismatch;
        for (0..n) |batch| {
            for (0..out_channels) |oc| {
                for (0..out_h) |y| {
                    for (0..out_w) |x_out| out[((batch * out_channels + oc) * out_h + y) * out_w + x_out] = bias_tensor.buffer.f32[oc];
                }
            }
        }
    }

    const in_channel_per_group = in_channels / group;
    for (0..n) |batch| {
        for (0..in_channels) |ic| {
            const group_index = ic / in_channel_per_group;
            for (0..in_h) |iy| {
                for (0..in_w) |ix| {
                    const input_value = x.buffer.f32[((batch * in_channels + ic) * in_h + iy) * in_w + ix];
                    for (0..out_channels_per_group) |oc_local| {
                        const oc = group_index * out_channels_per_group + oc_local;
                        for (0..kernel_h) |ky| {
                            const out_y_signed = @as(isize, @intCast(iy * strides.h + ky)) - @as(isize, @intCast(pads.top));
                            if (out_y_signed < 0 or out_y_signed >= @as(isize, @intCast(out_h))) continue;
                            for (0..kernel_w) |kx| {
                                const out_x_signed = @as(isize, @intCast(ix * strides.w + kx)) - @as(isize, @intCast(pads.left));
                                if (out_x_signed < 0 or out_x_signed >= @as(isize, @intCast(out_w))) continue;
                                const weight_index = ((ic * out_channels_per_group + oc_local) * kernel_h + ky) * kernel_w + kx;
                                const out_index = ((batch * out_channels + oc) * out_h + @as(usize, @intCast(out_y_signed))) * out_w + @as(usize, @intCast(out_x_signed));
                                out[out_index] += input_value * w.buffer.f32[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ n, out_channels, out_h, out_w }),
        .buffer = .{ .f32 = out },
    };
}

pub fn maxPool(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len != 4) return error.UnsupportedTensorRank;
    const kernel = try requiredPairAttribute(allocator, node, "kernel_shape");
    const strides = try pairAttribute(allocator, node, "strides", 1);
    const pads = try padsAttribute(allocator, node);
    const out_h = try convOutputDim(input.shape[2], kernel.h, pads.top, pads.bottom, strides.h);
    const out_w = try convOutputDim(input.shape[3], kernel.w, pads.left, pads.right, strides.w);
    const out = try allocator.alloc(f32, input.shape[0] * input.shape[1] * out_h * out_w);
    errdefer allocator.free(out);

    for (0..input.shape[0]) |batch| {
        for (0..input.shape[1]) |channel| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var max_value = -std.math.inf(f32);
                    for (0..kernel.h) |ky| {
                        const in_y = @as(isize, @intCast(oy * strides.h + ky)) - @as(isize, @intCast(pads.top));
                        if (in_y < 0 or in_y >= @as(isize, @intCast(input.shape[2]))) continue;
                        for (0..kernel.w) |kx| {
                            const in_x = @as(isize, @intCast(ox * strides.w + kx)) - @as(isize, @intCast(pads.left));
                            if (in_x < 0 or in_x >= @as(isize, @intCast(input.shape[3]))) continue;
                            const index = ((batch * input.shape[1] + channel) * input.shape[2] + @as(usize, @intCast(in_y))) * input.shape[3] + @as(usize, @intCast(in_x));
                            max_value = @max(max_value, input.buffer.f32[index]);
                        }
                    }
                    out[((batch * input.shape[1] + channel) * out_h + oy) * out_w + ox] = max_value;
                }
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ input.shape[0], input.shape[1], out_h, out_w }),
        .buffer = .{ .f32 = out },
    };
}

pub fn averagePool(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len != 4) return error.UnsupportedTensorRank;
    if ((common.attributeInt(node, "ceil_mode") orelse 0) != 0) return error.UnsupportedOperatorAttribute;
    const kernel = try requiredPairAttribute(allocator, node, "kernel_shape");
    const strides = try pairAttribute(allocator, node, "strides", 1);
    const pads = try padsAttribute(allocator, node);
    const count_include_pad = (common.attributeInt(node, "count_include_pad") orelse 0) != 0;
    const out_h = try convOutputDim(input.shape[2], kernel.h, pads.top, pads.bottom, strides.h);
    const out_w = try convOutputDim(input.shape[3], kernel.w, pads.left, pads.right, strides.w);
    const out = try allocator.alloc(f32, input.shape[0] * input.shape[1] * out_h * out_w);
    errdefer allocator.free(out);

    for (0..input.shape[0]) |batch| {
        for (0..input.shape[1]) |channel| {
            for (0..out_h) |oy| {
                for (0..out_w) |ox| {
                    var sum: f32 = 0;
                    var count: usize = 0;
                    for (0..kernel.h) |ky| {
                        const in_y = @as(isize, @intCast(oy * strides.h + ky)) - @as(isize, @intCast(pads.top));
                        if (in_y < 0 or in_y >= @as(isize, @intCast(input.shape[2]))) {
                            if (count_include_pad) count += kernel.w;
                            continue;
                        }
                        for (0..kernel.w) |kx| {
                            const in_x = @as(isize, @intCast(ox * strides.w + kx)) - @as(isize, @intCast(pads.left));
                            if (in_x < 0 or in_x >= @as(isize, @intCast(input.shape[3]))) {
                                if (count_include_pad) count += 1;
                                continue;
                            }
                            const index = ((batch * input.shape[1] + channel) * input.shape[2] + @as(usize, @intCast(in_y))) * input.shape[3] + @as(usize, @intCast(in_x));
                            sum += input.buffer.f32[index];
                            count += 1;
                        }
                    }
                    out[((batch * input.shape[1] + channel) * out_h + oy) * out_w + ox] = if (count == 0) 0 else sum / @as(f32, @floatFromInt(count));
                }
            }
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ input.shape[0], input.shape[1], out_h, out_w }),
        .buffer = .{ .f32 = out },
    };
}

pub fn globalAveragePool(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len != 4) return error.UnsupportedTensorRank;
    const h = input.shape[2];
    const w = input.shape[3];
    if (h == 0 or w == 0) return error.InvalidOperatorAttribute;

    const out = try allocator.alloc(f32, input.shape[0] * input.shape[1]);
    errdefer allocator.free(out);
    const denom = @as(f32, @floatFromInt(h * w));
    for (0..input.shape[0]) |batch| {
        for (0..input.shape[1]) |channel| {
            var sum: f32 = 0;
            for (0..h) |y| {
                for (0..w) |x| {
                    const index = ((batch * input.shape[1] + channel) * h + y) * w + x;
                    sum += input.buffer.f32[index];
                }
            }
            out[batch * input.shape[1] + channel] = sum / denom;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ input.shape[0], input.shape[1], 1, 1 }),
        .buffer = .{ .f32 = out },
    };
}

pub fn globalMaxPool(allocator: std.mem.Allocator, inputs: []const *const Tensor) !Tensor {
    if (inputs.len != 1) return error.InvalidOperatorArity;
    const input = inputs[0].*;
    if (input.buffer != .f32) return error.UnsupportedTensorDType;
    if (input.shape.len != 4) return error.UnsupportedTensorRank;
    const h = input.shape[2];
    const w = input.shape[3];
    if (h == 0 or w == 0) return error.InvalidOperatorAttribute;

    const out = try allocator.alloc(f32, input.shape[0] * input.shape[1]);
    errdefer allocator.free(out);
    for (0..input.shape[0]) |batch| {
        for (0..input.shape[1]) |channel| {
            var max_value = -std.math.inf(f32);
            for (0..h) |y| {
                for (0..w) |x| {
                    const index = ((batch * input.shape[1] + channel) * h + y) * w + x;
                    max_value = @max(max_value, input.buffer.f32[index]);
                }
            }
            out[batch * input.shape[1] + channel] = max_value;
        }
    }
    return .{
        .allocator = allocator,
        .shape = try allocator.dupe(usize, &.{ input.shape[0], input.shape[1], 1, 1 }),
        .buffer = .{ .f32 = out },
    };
}

const Pair = struct {
    h: usize,
    w: usize,
};

const Pads = struct {
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
};

fn convOutputDim(input: usize, kernel: usize, pad_begin: usize, pad_end: usize, stride: usize) !usize {
    if (kernel == 0 or stride == 0) return error.InvalidOperatorAttribute;
    const padded = input + pad_begin + pad_end;
    if (padded < kernel) return error.InvalidOutputShape;
    return ((padded - kernel) / stride) + 1;
}

fn convTransposeOutputDim(input: usize, kernel: usize, pad_begin: usize, pad_end: usize, stride: usize) !usize {
    if (kernel == 0 or stride == 0) return error.InvalidOperatorAttribute;
    const expanded = (input - 1) * stride + kernel;
    if (expanded < pad_begin + pad_end) return error.InvalidOutputShape;
    return expanded - pad_begin - pad_end;
}

fn pairAttribute(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, name: []const u8, default: usize) !Pair {
    const values = try common.attributeIntsOwned(allocator, node, name);
    defer allocator.free(values);
    if (values.len == 0) return .{ .h = default, .w = default };
    if (values.len != 2 or values[0] <= 0 or values[1] <= 0) return error.InvalidOperatorAttribute;
    return .{ .h = @intCast(values[0]), .w = @intCast(values[1]) };
}

fn requiredPairAttribute(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo, name: []const u8) !Pair {
    const values = try common.attributeIntsOwned(allocator, node, name);
    defer allocator.free(values);
    if (values.len != 2 or values[0] <= 0 or values[1] <= 0) return error.MissingOperatorAttribute;
    return .{ .h = @intCast(values[0]), .w = @intCast(values[1]) };
}

fn padsAttribute(allocator: std.mem.Allocator, node: onnx_metadata.NodeInfo) !Pads {
    const values = try common.attributeIntsOwned(allocator, node, "pads");
    defer allocator.free(values);
    if (values.len == 0) return .{ .top = 0, .left = 0, .bottom = 0, .right = 0 };
    if (values.len == 2) {
        if (values[0] < 0 or values[1] < 0) return error.InvalidOperatorAttribute;
        return .{ .top = @intCast(values[0]), .left = @intCast(values[1]), .bottom = @intCast(values[0]), .right = @intCast(values[1]) };
    }
    if (values.len == 4) {
        for (values) |value| if (value < 0) return error.InvalidOperatorAttribute;
        return .{ .top = @intCast(values[0]), .left = @intCast(values[1]), .bottom = @intCast(values[2]), .right = @intCast(values[3]) };
    }
    return error.InvalidOperatorAttribute;
}
