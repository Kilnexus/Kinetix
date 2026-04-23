const std = @import("std");
const onnx_metadata = @import("../onnx/metadata.zig");
const tensor_mod = @import("tensor.zig");
const op_registry = @import("../../ops/index.zig").registry;

const activation = @import("ops/activation.zig");
const core = @import("ops/core.zig");
const indexing = @import("ops/indexing.zig");
const linear = @import("ops/linear.zig");
const normalization = @import("ops/normalization.zig");
const shape = @import("ops/shape.zig");
const spatial = @import("ops/spatial.zig");

pub const Tensor = tensor_mod.Tensor;

pub fn isSupported(op_type: []const u8) bool {
    return op_registry.isGraphExecutableOnnx(op_type);
}

pub fn execute(
    allocator: std.mem.Allocator,
    node: onnx_metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    if (std.mem.eql(u8, node.op_type, "Constant")) return try core.constant(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Identity")) return try core.identity(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Add")) return try core.elementwise(allocator, inputs, .add);
    if (std.mem.eql(u8, node.op_type, "Mul")) return try core.elementwise(allocator, inputs, .mul);
    if (std.mem.eql(u8, node.op_type, "Relu")) return try core.relu(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Sigmoid")) return try activation.sigmoid(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "LeakyRelu")) return try activation.leakyRelu(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Gelu")) return try activation.gelu(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "SwiGLU")) return try activation.swiglu(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Cast")) return try core.cast(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "MatMul")) return try linear.matmul(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Gemm")) return try linear.gemm(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Reshape")) return try shape.reshape(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Shape")) return try shape.shapeOp(allocator, inputs);
    if (std.mem.eql(u8, node.op_type, "Unsqueeze")) return try shape.unsqueeze(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Squeeze")) return try shape.squeeze(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Concat")) return try shape.concat(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Transpose")) return try shape.transpose(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Gather")) return try indexing.gather(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Slice")) return try indexing.slice(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Softmax")) return try normalization.softmax(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "LayerNormalization")) return try normalization.layerNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "RMSNormalization")) return try normalization.rmsNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "Conv")) return try spatial.conv(allocator, node, inputs);
    if (std.mem.eql(u8, node.op_type, "MaxPool")) return try spatial.maxPool(allocator, node, inputs);
    return error.UnsupportedOnnxOperator;
}

test "runtime ops execute f32 matmul and relu" {
    var lhs = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer lhs.deinit();
    var rhs = try Tensor.fromF32(std.testing.allocator, &.{ 2, 1 }, &.{ 10, 20 });
    defer rhs.deinit();
    const matmul_node = testNode("MatMul", &.{});
    var out = try execute(std.testing.allocator, matmul_node, &.{ &lhs, &rhs });
    defer out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 110 }, out.buffer.f32);

    const relu_node = testNode("Relu", &.{});
    var relu_out = try execute(std.testing.allocator, relu_node, &.{&out});
    defer relu_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 50, 110 }, relu_out.buffer.f32);
}

test "runtime ops execute shape cast and gather" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    const shape_node = testNode("Shape", &.{});
    var shape_out = try execute(std.testing.allocator, shape_node, &.{&input});
    defer shape_out.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, shape_out.buffer.i64);

    var to_f32_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("to", 1)};
    const cast_node = testNode("Cast", to_f32_attr[0..]);
    var cast_out = try execute(std.testing.allocator, cast_node, &.{&shape_out});
    defer cast_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, cast_out.buffer.f32);

    var data = try Tensor.fromI64(std.testing.allocator, &.{3}, &.{ 10, 20, 30 });
    defer data.deinit();
    var indices = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 2, 0 });
    defer indices.deinit();
    var axis_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("axis", 0)};
    const gather_node = testNode("Gather", axis_attr[0..]);
    var gather_out = try execute(std.testing.allocator, gather_node, &.{ &data, &indices });
    defer gather_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{2}, gather_out.shape);
    try std.testing.expectEqualSlices(i64, &.{ 30, 10 }, gather_out.buffer.i64);
}

test "runtime ops execute unsqueeze squeeze concat and transpose" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer input.deinit();
    var unsqueeze_axes_values = [_]i64{0};
    var unsqueeze_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("axes", unsqueeze_axes_values[0..])};
    const unsqueeze_node = testNode("Unsqueeze", unsqueeze_attrs[0..]);
    var unsqueezed = try execute(std.testing.allocator, unsqueeze_node, &.{&input});
    defer unsqueezed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, unsqueezed.shape);

    var squeeze_axes_values = [_]i64{0};
    var squeeze_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("axes", squeeze_axes_values[0..])};
    const squeeze_node = testNode("Squeeze", squeeze_attrs[0..]);
    var squeezed = try execute(std.testing.allocator, squeeze_node, &.{&unsqueezed});
    defer squeezed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, squeezed.shape);
    try std.testing.expectEqualSlices(f32, input.buffer.f32, squeezed.buffer.f32);

    var left = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer left.deinit();
    var right = try Tensor.fromI64(std.testing.allocator, &.{3}, &.{ 3, 4, 5 });
    defer right.deinit();
    var concat_axis_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("axis", 0)};
    const concat_node = testNode("Concat", concat_axis_attr[0..]);
    var concat_out = try execute(std.testing.allocator, concat_node, &.{ &left, &right });
    defer concat_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{5}, concat_out.shape);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 5 }, concat_out.buffer.i64);

    var perm_values = [_]i64{ 1, 0 };
    var transpose_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("perm", perm_values[0..])};
    const transpose_node = testNode("Transpose", transpose_attrs[0..]);
    var transposed = try execute(std.testing.allocator, transpose_node, &.{&input});
    defer transposed.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, transposed.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, transposed.buffer.f32);
}

test "runtime ops execute slice softmax gemm and layer normalization" {
    var matrix = try Tensor.fromF32(std.testing.allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer matrix.deinit();
    var starts = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 0, 1 });
    defer starts.deinit();
    var ends = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 2, 3 });
    defer ends.deinit();
    var axes = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 0, 1 });
    defer axes.deinit();
    const slice_node = testNode("Slice", &.{});
    var sliced = try execute(std.testing.allocator, slice_node, &.{ &matrix, &starts, &ends, &axes });
    defer sliced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, sliced.shape);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 5, 6 }, sliced.buffer.f32);

    var softmax_axis_attr = [_]onnx_metadata.AttributeInfo{testIntAttribute("axis", 1)};
    const softmax_node = testNode("Softmax", softmax_axis_attr[0..]);
    var softmax_out = try execute(std.testing.allocator, softmax_node, &.{&matrix});
    defer softmax_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), softmax_out.buffer.f32[0] + softmax_out.buffer.f32[1] + softmax_out.buffer.f32[2], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), softmax_out.buffer.f32[3] + softmax_out.buffer.f32[4] + softmax_out.buffer.f32[5], 0.00001);

    var weights = try Tensor.fromF32(std.testing.allocator, &.{ 3, 2 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer weights.deinit();
    var bias = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 10, 20 });
    defer bias.deinit();
    var gemm_attrs = [_]onnx_metadata.AttributeInfo{testFloatAttribute("alpha", 1)};
    const gemm_node = testNode("Gemm", gemm_attrs[0..]);
    var gemm_out = try execute(std.testing.allocator, gemm_node, &.{ &matrix, &weights, &bias });
    defer gemm_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, gemm_out.shape);
    try std.testing.expectEqualSlices(f32, &.{ 32, 48, 59, 84 }, gemm_out.buffer.f32);

    var scale = try Tensor.fromF32(std.testing.allocator, &.{3}, &.{ 1, 1, 1 });
    defer scale.deinit();
    var ln_bias = try Tensor.fromF32(std.testing.allocator, &.{3}, &.{ 0, 0, 0 });
    defer ln_bias.deinit();
    var layer_attrs = [_]onnx_metadata.AttributeInfo{
        testIntAttribute("axis", 1),
        testFloatAttribute("epsilon", 0.00001),
    };
    const layer_node = testNode("LayerNormalization", layer_attrs[0..]);
    var normalized = try execute(std.testing.allocator, layer_node, &.{ &matrix, &scale, &ln_bias });
    defer normalized.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -1.2247356), normalized.buffer.f32[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), normalized.buffer.f32[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2247356), normalized.buffer.f32[2], 0.00001);
}

test "runtime ops execute collected activation and rms normalization adapters" {
    var values = try Tensor.fromF32(std.testing.allocator, &.{3}, &.{ -1, 0, 1 });
    defer values.deinit();

    const sigmoid_node = testNode("Sigmoid", &.{});
    var sigmoid_out = try execute(std.testing.allocator, sigmoid_node, &.{&values});
    defer sigmoid_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), sigmoid_out.buffer.f32[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid_out.buffer.f32[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), sigmoid_out.buffer.f32[2], 0.00001);

    var leaky_attrs = [_]onnx_metadata.AttributeInfo{testFloatAttribute("alpha", 0.2)};
    const leaky_node = testNode("LeakyRelu", leaky_attrs[0..]);
    var leaky_out = try execute(std.testing.allocator, leaky_node, &.{&values});
    defer leaky_out.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -0.2, 0, 1 }, leaky_out.buffer.f32);

    const gelu_node = testNode("Gelu", &.{});
    var gelu_out = try execute(std.testing.allocator, gelu_node, &.{&values});
    defer gelu_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -0.158808), gelu_out.buffer.f32[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), gelu_out.buffer.f32[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.841192), gelu_out.buffer.f32[2], 0.00001);

    var matrix = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 3, 4, 6, 8 });
    defer matrix.deinit();
    var scale = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer scale.deinit();
    var rms_attrs = [_]onnx_metadata.AttributeInfo{
        testIntAttribute("axis", 1),
        testFloatAttribute("epsilon", 0),
    };
    const rms_node = testNode("RMSNormalization", rms_attrs[0..]);
    var rms_out = try execute(std.testing.allocator, rms_node, &.{ &matrix, &scale });
    defer rms_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.84852814), rms_out.buffer.f32[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.2627418), rms_out.buffer.f32[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.84852814), rms_out.buffer.f32[2], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.2627418), rms_out.buffer.f32[3], 0.00001);
}

test "runtime ops execute collected swiglu and spatial adapters" {
    var gate = try Tensor.fromF32(std.testing.allocator, &.{3}, &.{ 0, 1, -1 });
    defer gate.deinit();
    var up = try Tensor.fromF32(std.testing.allocator, &.{3}, &.{ 1, 2, 3 });
    defer up.deinit();
    const swiglu_node = testNode("SwiGLU", &.{});
    var swiglu_out = try execute(std.testing.allocator, swiglu_node, &.{ &gate, &up });
    defer swiglu_out.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0), swiglu_out.buffer.f32[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.4621172), swiglu_out.buffer.f32[1], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8068243), swiglu_out.buffer.f32[2], 0.00001);

    var pool_input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 1, 2, 2 }, &.{ 1, 9, 3, 4 });
    defer pool_input.deinit();
    var pool_kernel_values = [_]i64{ 2, 2 };
    var pool_stride_values = [_]i64{ 2, 2 };
    var pool_attrs = [_]onnx_metadata.AttributeInfo{
        testIntsAttribute("kernel_shape", pool_kernel_values[0..]),
        testIntsAttribute("strides", pool_stride_values[0..]),
    };
    const pool_node = testNode("MaxPool", pool_attrs[0..]);
    var pool_out = try execute(std.testing.allocator, pool_node, &.{&pool_input});
    defer pool_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, pool_out.shape);
    try std.testing.expectEqualSlices(f32, &.{9}, pool_out.buffer.f32);

    var conv_input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2, 1, 1 }, &.{ 2, 3 });
    defer conv_input.deinit();
    var conv_weights = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2, 1, 1 }, &.{ 4, 5 });
    defer conv_weights.deinit();
    var conv_bias = try Tensor.fromF32(std.testing.allocator, &.{1}, &.{1});
    defer conv_bias.deinit();
    var conv_stride_values = [_]i64{ 1, 1 };
    var conv_attrs = [_]onnx_metadata.AttributeInfo{testIntsAttribute("strides", conv_stride_values[0..])};
    const conv_node = testNode("Conv", conv_attrs[0..]);
    var conv_out = try execute(std.testing.allocator, conv_node, &.{ &conv_input, &conv_weights, &conv_bias });
    defer conv_out.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, conv_out.shape);
    try std.testing.expectEqualSlices(f32, &.{24}, conv_out.buffer.f32);
}

fn testNode(op_type: []const u8, attributes: []onnx_metadata.AttributeInfo) onnx_metadata.NodeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(""),
        .op_type = @constCast(op_type),
        .domain = @constCast(""),
        .attributes = attributes,
    };
}

fn testIntAttribute(name: []const u8, value: i64) onnx_metadata.AttributeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(name),
        .int_value = value,
        .int_count = 1,
    };
}

fn testFloatAttribute(name: []const u8, value: f32) onnx_metadata.AttributeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(name),
        .float_value = value,
        .float_count = 1,
    };
}

fn testIntsAttribute(name: []const u8, values: []i64) onnx_metadata.AttributeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast(name),
        .int_values = values,
        .int_count = values.len,
    };
}
