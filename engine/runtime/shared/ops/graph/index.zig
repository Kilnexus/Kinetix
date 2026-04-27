const std = @import("std");
const shared_graph = @import("shared_graph");
const op_registry = @import("../registry.zig");

pub const common = @import("common.zig");
pub const activation = @import("activation/index.zig");
pub const core = @import("core/index.zig");
pub const indexing = @import("indexing/index.zig");
pub const linear = @import("linear/index.zig");
pub const normalization = @import("normalization/index.zig");
pub const shape = @import("shape/index.zig");
pub const spatial = @import("spatial/index.zig");

pub const Tensor = shared_graph.runtime.tensor.Tensor;

pub fn isSupported(op_type: []const u8) bool {
    return op_registry.isGraphExecutableOnnx(op_type);
}

pub fn execute(
    allocator: std.mem.Allocator,
    node: shared_graph.onnx.metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    const entry = op_registry.findGraphExecutableOnnx(node.op_type) orelse return error.UnsupportedOnnxOperator;
    const kernel_abi = entry.kernelAbi();
    if (kernel_abi.class != .graph_op) return error.UnsupportedOnnxOperator;

    if (std.mem.eql(u8, entry.name, "Constant")) return try core.constant(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Identity")) return try core.identity(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Add")) return try core.elementwise(allocator, inputs, .add);
    if (std.mem.eql(u8, entry.name, "Sub")) return try core.elementwise(allocator, inputs, .sub);
    if (std.mem.eql(u8, entry.name, "Mul")) return try core.elementwise(allocator, inputs, .mul);
    if (std.mem.eql(u8, entry.name, "Div")) return try core.elementwise(allocator, inputs, .div);
    if (std.mem.eql(u8, entry.name, "Relu")) return try core.relu(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Sigmoid")) return try activation.sigmoid(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Tanh")) return try activation.tanh(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "HardSwish")) return try activation.hardSwish(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "LeakyRelu")) return try activation.leakyRelu(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Gelu")) return try activation.gelu(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "SwiGLU")) return try activation.swiglu(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Cast")) return try core.cast(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "MatMul")) return try linear.matmul(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Gemm")) return try linear.gemm(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Reshape")) return try shape.reshape(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Flatten")) return try shape.flatten(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Shape")) return try shape.shapeOp(allocator, inputs);
    if (std.mem.eql(u8, entry.name, "Unsqueeze")) return try shape.unsqueeze(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Squeeze")) return try shape.squeeze(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Concat")) return try shape.concat(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Transpose")) return try shape.transpose(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Gather")) return try indexing.gather(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "ArgMax")) return try indexing.argMax(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Slice")) return try indexing.slice(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Softmax")) return try normalization.softmax(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "ReduceMean")) return try normalization.reduceMean(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "BatchNormalization")) return try normalization.batchNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "LayerNormalization")) return try normalization.layerNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "RMSNormalization")) return try normalization.rmsNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "Conv")) return try spatial.conv(allocator, node, inputs);
    if (std.mem.eql(u8, entry.name, "MaxPool")) return try spatial.maxPool(allocator, node, inputs);
    return error.UnsupportedOnnxOperator;
}

test "graph dispatcher executes registry-backed arithmetic ops" {
    var lhs = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 6, 8 });
    defer lhs.deinit();
    var rhs = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 2, 4 });
    defer rhs.deinit();
    const inputs = [_]*const Tensor{ &lhs, &rhs };

    var sub = try execute(std.testing.allocator, testNode("Sub"), &inputs);
    defer sub.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 4 }, sub.buffer.f32);

    var div = try execute(std.testing.allocator, testNode("Div"), &inputs);
    defer div.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2 }, div.buffer.f32);
}

test "graph dispatcher executes registry-backed tanh op" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1 }, &.{0});
    defer input.deinit();
    const inputs = [_]*const Tensor{&input};

    var output = try execute(std.testing.allocator, testNode("Tanh"), &inputs);
    defer output.deinit();
    try std.testing.expectEqual(@as(f32, 0), output.buffer.f32[0]);
}

test "graph dispatcher executes paddleocr common ops" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2, 2 }, &.{ 1, 3, 2, 4 });
    defer input.deinit();
    const single_input = [_]*const Tensor{&input};

    var flat = try execute(std.testing.allocator, testNode("Flatten"), &single_input);
    defer flat.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, flat.shape);

    var argmax = try execute(std.testing.allocator, testNode("ArgMax"), &single_input);
    defer argmax.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0 }, argmax.buffer.i64);

    var reduced = try execute(std.testing.allocator, testNode("ReduceMean"), &single_input);
    defer reduced.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1 }, reduced.shape);
    try std.testing.expectEqual(@as(f32, 2.5), reduced.buffer.f32[0]);

    var hswish = try execute(std.testing.allocator, testNode("HardSwish"), &single_input);
    defer hswish.deinit();
    try std.testing.expect(hswish.buffer.f32[0] > 0);
}

test "graph dispatcher executes batch normalization" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2, 1, 1 }, &.{ 2, 4 });
    defer input.deinit();
    var scale = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 1, 1 });
    defer scale.deinit();
    var bias = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 0, 0 });
    defer bias.deinit();
    var mean = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 1, 2 });
    defer mean.deinit();
    var variance = try Tensor.fromF32(std.testing.allocator, &.{2}, &.{ 1, 4 });
    defer variance.deinit();
    const inputs = [_]*const Tensor{ &input, &scale, &bias, &mean, &variance };

    var output = try execute(std.testing.allocator, testNode("BatchNormalization"), &inputs);
    defer output.deinit();
    try std.testing.expect(output.buffer.f32[0] > 0.99 and output.buffer.f32[0] < 1.01);
}

fn testNode(op_type: []const u8) shared_graph.onnx.metadata.NodeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast("test"),
        .op_type = @constCast(op_type),
        .domain = @constCast(""),
    };
}
