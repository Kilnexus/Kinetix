const std = @import("std");
const metadata = @import("../onnx/metadata.zig");
const tensor_mod = @import("../runtime/tensor.zig");
const external = @import("external.zig");

pub const Tensor = tensor_mod.Tensor;

pub fn materialize(
    allocator: std.mem.Allocator,
    info: metadata.TensorInfo,
    external_store: ?*external.Store,
) !Tensor {
    if (info.isExternal()) {
        const store = external_store orelse return error.TensorIsExternal;
        const bytes = try store.tensorBytes(info);
        return try materializeBytes(allocator, info, bytes.bytes);
    }
    if (info.raw_data.len != 0) {
        return try materializeBytes(allocator, info, info.raw_data);
    }
    return try materializeTypedValues(allocator, info);
}

pub fn materializeBytes(
    allocator: std.mem.Allocator,
    info: metadata.TensorInfo,
    bytes: []const u8,
) !Tensor {
    const shape = try shapeFromDims(allocator, info.dims);
    defer allocator.free(shape);
    const expected_count = elementCountFromDims(info.dims) catch return error.DynamicTensorShape;

    return switch (info.elem_type.raw) {
        1 => blk: {
            const values = try rawF32Owned(allocator, bytes, expected_count);
            defer allocator.free(values);
            break :blk try Tensor.fromF32(allocator, shape, values);
        },
        6 => blk: {
            const values = try rawI32Owned(allocator, bytes, expected_count);
            defer allocator.free(values);
            break :blk try Tensor.fromI32(allocator, shape, values);
        },
        7 => blk: {
            const values = try rawI64Owned(allocator, bytes, expected_count);
            defer allocator.free(values);
            break :blk try Tensor.fromI64(allocator, shape, values);
        },
        9 => blk: {
            const values = try rawBoolAsI64Owned(allocator, bytes, expected_count);
            defer allocator.free(values);
            break :blk try Tensor.fromI64(allocator, shape, values);
        },
        else => error.UnsupportedOnnxElementType,
    };
}

fn materializeTypedValues(allocator: std.mem.Allocator, info: metadata.TensorInfo) !Tensor {
    const shape = try shapeFromDims(allocator, info.dims);
    defer allocator.free(shape);
    return switch (info.elem_type.raw) {
        1 => try Tensor.fromF32(allocator, shape, info.float_data),
        6 => try Tensor.fromI32(allocator, shape, info.int32_data),
        7 => try Tensor.fromI64(allocator, shape, info.int64_data),
        9 => blk: {
            const values = try boolTypedAsI64Owned(allocator, info);
            defer allocator.free(values);
            break :blk try Tensor.fromI64(allocator, shape, values);
        },
        else => error.UnsupportedOnnxElementType,
    };
}

fn rawF32Owned(allocator: std.mem.Allocator, bytes: []const u8, expected_count: usize) ![]f32 {
    if (bytes.len != expected_count * 4) return error.TensorElementCountMismatch;
    const out = try allocator.alloc(f32, expected_count);
    errdefer allocator.free(out);
    for (out, 0..) |*slot, index| {
        const raw = bytes[index * 4 ..][0..4];
        slot.* = @bitCast(std.mem.readInt(u32, raw, .little));
    }
    return out;
}

fn rawI32Owned(allocator: std.mem.Allocator, bytes: []const u8, expected_count: usize) ![]i32 {
    if (bytes.len != expected_count * 4) return error.TensorElementCountMismatch;
    const out = try allocator.alloc(i32, expected_count);
    errdefer allocator.free(out);
    for (out, 0..) |*slot, index| {
        const raw = bytes[index * 4 ..][0..4];
        slot.* = @bitCast(std.mem.readInt(u32, raw, .little));
    }
    return out;
}

fn rawI64Owned(allocator: std.mem.Allocator, bytes: []const u8, expected_count: usize) ![]i64 {
    if (bytes.len != expected_count * 8) return error.TensorElementCountMismatch;
    const out = try allocator.alloc(i64, expected_count);
    errdefer allocator.free(out);
    for (out, 0..) |*slot, index| {
        const raw = bytes[index * 8 ..][0..8];
        slot.* = @bitCast(std.mem.readInt(u64, raw, .little));
    }
    return out;
}

fn rawBoolAsI64Owned(allocator: std.mem.Allocator, bytes: []const u8, expected_count: usize) ![]i64 {
    if (bytes.len != expected_count) return error.TensorElementCountMismatch;
    const out = try allocator.alloc(i64, expected_count);
    errdefer allocator.free(out);
    for (bytes, out) |byte, *slot| slot.* = if (byte == 0) 0 else 1;
    return out;
}

fn boolTypedAsI64Owned(allocator: std.mem.Allocator, info: metadata.TensorInfo) ![]i64 {
    const expected_count = try elementCountFromDims(info.dims);
    const out = try allocator.alloc(i64, expected_count);
    errdefer allocator.free(out);
    if (info.int32_data.len == expected_count) {
        for (info.int32_data, out) |value, *slot| slot.* = if (value == 0) 0 else 1;
        return out;
    }
    if (info.int64_data.len == expected_count) {
        for (info.int64_data, out) |value, *slot| slot.* = if (value == 0) 0 else 1;
        return out;
    }
    return error.TensorElementCountMismatch;
}

fn elementCountFromDims(dims: []const metadata.Dimension) !usize {
    var total: usize = 1;
    for (dims) |dim| {
        const value = switch (dim) {
            .value => |item| item,
            else => return error.DynamicTensorShape,
        };
        if (value < 0) return error.DynamicTensorShape;
        total = try std.math.mul(usize, total, @intCast(value));
    }
    return total;
}

fn shapeFromDims(allocator: std.mem.Allocator, dims: []const metadata.Dimension) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    errdefer allocator.free(shape);
    for (dims, shape) |dim, *slot| {
        const value = switch (dim) {
            .value => |item| item,
            else => return error.DynamicTensorShape,
        };
        if (value < 0) return error.DynamicTensorShape;
        slot.* = @intCast(value);
    }
    return shape;
}

test "onnx initializer materializes raw f32 tensor" {
    const dims = [_]metadata.Dimension{ .{ .value = 2 } };
    const bytes = [_]u8{ 0, 0, 128, 63, 0, 0, 0, 64 };
    const info = metadata.TensorInfo{
        .allocator = std.testing.allocator,
        .name = @constCast("w"),
        .elem_type = .{ .raw = 1 },
        .dims = &dims,
        .raw_data = @constCast(bytes[0..]),
        .raw_data_len = bytes.len,
    };

    var tensor = try materialize(std.testing.allocator, info, null);
    defer tensor.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, tensor.buffer.f32);
}

test "onnx initializer materializes typed int64 tensor" {
    const dims = [_]metadata.Dimension{ .{ .value = 2 } };
    const values = [_]i64{ 4, 8 };
    const info = metadata.TensorInfo{
        .allocator = std.testing.allocator,
        .name = @constCast("shape"),
        .elem_type = .{ .raw = 7 },
        .dims = &dims,
        .int64_data = @constCast(values[0..]),
    };

    var tensor = try materialize(std.testing.allocator, info, null);
    defer tensor.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 4, 8 }, tensor.buffer.i64);
}
