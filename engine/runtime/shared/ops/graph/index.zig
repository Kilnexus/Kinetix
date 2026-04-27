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

pub const ExecutionOutputs = struct {
    allocator: std.mem.Allocator,
    tensors: []Tensor,

    pub fn deinit(self: *ExecutionOutputs) void {
        for (self.tensors) |*tensor| tensor.deinit();
        self.allocator.free(self.tensors);
        self.* = undefined;
    }
};

pub fn isSupported(op_type: []const u8) bool {
    return op_registry.isGraphExecutableOnnx(op_type);
}

pub fn executeAll(
    allocator: std.mem.Allocator,
    node: shared_graph.onnx.metadata.NodeInfo,
    inputs: []const *const Tensor,
) !ExecutionOutputs {
    const entry = op_registry.findGraphExecutableOnnx(node.op_type) orelse return error.UnsupportedOnnxOperator;
    const kernel_abi = entry.kernelAbi();
    if (kernel_abi.class != .graph_op) return error.UnsupportedOnnxOperator;

    if (std.mem.eql(u8, entry.name, "Split")) {
        return .{ .allocator = allocator, .tensors = try shape.splitAll(allocator, node, inputs) };
    }
    if (std.mem.eql(u8, entry.name, "TopK")) {
        return .{ .allocator = allocator, .tensors = try indexing.topKAll(allocator, node, inputs) };
    }

    var tensor = try executeSingle(allocator, entry.name, node, inputs);
    errdefer tensor.deinit();
    return try singleOutput(allocator, tensor);
}

pub fn execute(
    allocator: std.mem.Allocator,
    node: shared_graph.onnx.metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    var outputs = try executeAll(allocator, node, inputs);
    if (outputs.tensors.len == 0) {
        outputs.deinit();
        return error.InvalidOperatorArity;
    }
    const first = outputs.tensors[0];
    for (outputs.tensors[1..]) |*tensor| tensor.deinit();
    allocator.free(outputs.tensors);
    return first;
}

fn executeSingle(
    allocator: std.mem.Allocator,
    name: []const u8,
    node: shared_graph.onnx.metadata.NodeInfo,
    inputs: []const *const Tensor,
) !Tensor {
    if (std.mem.eql(u8, name, "Constant")) return try core.constant(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Identity")) return try core.identity(allocator, inputs);
    if (std.mem.eql(u8, name, "Add")) return try core.elementwise(allocator, inputs, .add);
    if (std.mem.eql(u8, name, "Sub")) return try core.elementwise(allocator, inputs, .sub);
    if (std.mem.eql(u8, name, "Mul")) return try core.elementwise(allocator, inputs, .mul);
    if (std.mem.eql(u8, name, "Div")) return try core.elementwise(allocator, inputs, .div);
    if (std.mem.eql(u8, name, "Min")) return try core.elementwise(allocator, inputs, .min);
    if (std.mem.eql(u8, name, "Max")) return try core.elementwise(allocator, inputs, .max);
    if (std.mem.eql(u8, name, "Pow")) return try core.elementwise(allocator, inputs, .pow);
    if (std.mem.eql(u8, name, "Relu")) return try core.relu(allocator, inputs);
    if (std.mem.eql(u8, name, "Clip")) return try core.clip(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Equal")) return try core.compare(allocator, inputs, .equal);
    if (std.mem.eql(u8, name, "Greater")) return try core.compare(allocator, inputs, .greater);
    if (std.mem.eql(u8, name, "Less")) return try core.compare(allocator, inputs, .less);
    if (std.mem.eql(u8, name, "And")) return try core.logical(allocator, inputs, .and_op);
    if (std.mem.eql(u8, name, "Or")) return try core.logical(allocator, inputs, .or_op);
    if (std.mem.eql(u8, name, "Not")) return try core.notOp(allocator, inputs);
    if (std.mem.eql(u8, name, "Floor")) return try core.unaryFloat(allocator, inputs, .floor);
    if (std.mem.eql(u8, name, "Ceil")) return try core.unaryFloat(allocator, inputs, .ceil_op);
    if (std.mem.eql(u8, name, "Sqrt")) return try core.unaryFloat(allocator, inputs, .sqrt);
    if (std.mem.eql(u8, name, "Exp")) return try core.unaryFloat(allocator, inputs, .exp);
    if (std.mem.eql(u8, name, "Log")) return try core.unaryFloat(allocator, inputs, .log);
    if (std.mem.eql(u8, name, "Range")) return try core.range(allocator, inputs);
    if (std.mem.eql(u8, name, "Sigmoid")) return try activation.sigmoid(allocator, inputs);
    if (std.mem.eql(u8, name, "Tanh")) return try activation.tanh(allocator, inputs);
    if (std.mem.eql(u8, name, "HardSwish")) return try activation.hardSwish(allocator, inputs);
    if (std.mem.eql(u8, name, "LeakyRelu")) return try activation.leakyRelu(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Gelu")) return try activation.gelu(allocator, inputs);
    if (std.mem.eql(u8, name, "SwiGLU")) return try activation.swiglu(allocator, inputs);
    if (std.mem.eql(u8, name, "Cast")) return try core.cast(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Where")) return try core.whereOp(allocator, inputs);
    if (std.mem.eql(u8, name, "MatMul")) return try linear.matmul(allocator, inputs);
    if (std.mem.eql(u8, name, "Gemm")) return try linear.gemm(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Reshape")) return try shape.reshape(allocator, inputs);
    if (std.mem.eql(u8, name, "Flatten")) return try shape.flatten(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Shape")) return try shape.shapeOp(allocator, inputs);
    if (std.mem.eql(u8, name, "Resize")) return try shape.resize(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Pad")) return try shape.pad(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Expand")) return try shape.expand(allocator, inputs);
    if (std.mem.eql(u8, name, "Tile")) return try shape.tile(allocator, inputs);
    if (std.mem.eql(u8, name, "Split")) return try shape.split(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Unsqueeze")) return try shape.unsqueeze(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Squeeze")) return try shape.squeeze(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Concat")) return try shape.concat(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Transpose")) return try shape.transpose(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Gather")) return try indexing.gather(allocator, node, inputs);
    if (std.mem.eql(u8, name, "ArgMax")) return try indexing.argMax(allocator, node, inputs);
    if (std.mem.eql(u8, name, "NonZero")) return try indexing.nonZero(allocator, inputs);
    if (std.mem.eql(u8, name, "TopK")) return try indexing.topKOutput(allocator, node, inputs, @intCast(common.attributeInt(node, "kinetix_output_index") orelse 0));
    if (std.mem.eql(u8, name, "Slice")) return try indexing.slice(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Softmax")) return try normalization.softmax(allocator, node, inputs);
    if (std.mem.eql(u8, name, "ReduceMean")) return try normalization.reduceMean(allocator, node, inputs);
    if (std.mem.eql(u8, name, "ReduceSum")) return try normalization.reduceSum(allocator, node, inputs);
    if (std.mem.eql(u8, name, "ReduceMax")) return try normalization.reduceMax(allocator, node, inputs);
    if (std.mem.eql(u8, name, "BatchNormalization")) return try normalization.batchNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, name, "LayerNormalization")) return try normalization.layerNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, name, "RMSNormalization")) return try normalization.rmsNormalization(allocator, node, inputs);
    if (std.mem.eql(u8, name, "Conv")) return try spatial.conv(allocator, node, inputs);
    if (std.mem.eql(u8, name, "ConvTranspose")) return try spatial.convTranspose(allocator, node, inputs);
    if (std.mem.eql(u8, name, "MaxPool")) return try spatial.maxPool(allocator, node, inputs);
    if (std.mem.eql(u8, name, "AveragePool")) return try spatial.averagePool(allocator, node, inputs);
    if (std.mem.eql(u8, name, "GlobalAveragePool")) return try spatial.globalAveragePool(allocator, inputs);
    if (std.mem.eql(u8, name, "GlobalMaxPool")) return try spatial.globalMaxPool(allocator, inputs);
    return error.UnsupportedOnnxOperator;
}

fn singleOutput(allocator: std.mem.Allocator, tensor: Tensor) !ExecutionOutputs {
    const tensors = try allocator.alloc(Tensor, 1);
    tensors[0] = tensor;
    return .{ .allocator = allocator, .tensors = tensors };
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

    var pow = try execute(std.testing.allocator, testNode("Pow"), &inputs);
    defer pow.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 36, 4096 }, pow.buffer.f32);

    var min = try execute(std.testing.allocator, testNode("Min"), &inputs);
    defer min.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, min.buffer.f32);
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

    var clip = try execute(std.testing.allocator, testNode("Clip"), &single_input);
    defer clip.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 2, 4 }, clip.buffer.f32);
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

test "graph dispatcher executes paddleocr shape utility ops" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 2 }, &.{ 5, 6 });
    defer input.deinit();
    var pads = try Tensor.fromI64(std.testing.allocator, &.{4}, &.{ 0, 1, 0, 1 });
    defer pads.deinit();
    var expanded_shape = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 2, 2 });
    defer expanded_shape.deinit();
    var split_sizes = try Tensor.fromI64(std.testing.allocator, &.{2}, &.{ 1, 1 });
    defer split_sizes.deinit();

    const pad_inputs = [_]*const Tensor{ &input, &pads };
    var padded = try execute(std.testing.allocator, testNode("Pad"), &pad_inputs);
    defer padded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 4 }, padded.shape);
    try std.testing.expectEqualSlices(f32, &.{ 0, 5, 6, 0 }, padded.buffer.f32);

    const expand_inputs = [_]*const Tensor{ &input, &expanded_shape };
    var expanded = try execute(std.testing.allocator, testNode("Expand"), &expand_inputs);
    defer expanded.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 5, 6 }, expanded.buffer.f32);

    const split_inputs = [_]*const Tensor{ &input, &split_sizes };
    var split = try execute(std.testing.allocator, testNode("Split"), &split_inputs);
    defer split.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, split.shape);
    try std.testing.expectEqualSlices(f32, &.{5}, split.buffer.f32);

    var split_all = try executeAll(std.testing.allocator, testNode("Split"), &split_inputs);
    defer split_all.deinit();
    try std.testing.expectEqual(@as(usize, 2), split_all.tensors.len);
    try std.testing.expectEqualSlices(f32, &.{5}, split_all.tensors[0].buffer.f32);
    try std.testing.expectEqualSlices(f32, &.{6}, split_all.tensors[1].buffer.f32);
}

test "graph dispatcher executes where op" {
    var condition = try Tensor.fromI64(std.testing.allocator, &.{ 2 }, &.{ 1, 0 });
    defer condition.deinit();
    var x = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 3, 3 });
    defer x.deinit();
    var y = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 7, 7 });
    defer y.deinit();
    const inputs = [_]*const Tensor{ &condition, &x, &y };

    var output = try execute(std.testing.allocator, testNode("Where"), &inputs);
    defer output.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 7 }, output.buffer.f32);
}

test "graph dispatcher executes control and dynamic shape helpers" {
    var lhs = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 1.2, 3.8 });
    defer lhs.deinit();
    var rhs = try Tensor.fromF32(std.testing.allocator, &.{ 2 }, &.{ 1.2, 4.0 });
    defer rhs.deinit();
    const compare_inputs = [_]*const Tensor{ &lhs, &rhs };

    var eq = try execute(std.testing.allocator, testNode("Equal"), &compare_inputs);
    defer eq.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, eq.buffer.i64);

    var floor = try execute(std.testing.allocator, testNode("Floor"), &.{&lhs});
    defer floor.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3 }, floor.buffer.f32);

    var sqrt = try execute(std.testing.allocator, testNode("Sqrt"), &.{&rhs});
    defer sqrt.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), sqrt.buffer.f32[1], 0.0001);

    var start = try Tensor.fromI64(std.testing.allocator, &.{1}, &.{0});
    defer start.deinit();
    var limit = try Tensor.fromI64(std.testing.allocator, &.{1}, &.{5});
    defer limit.deinit();
    var delta = try Tensor.fromI64(std.testing.allocator, &.{1}, &.{2});
    defer delta.deinit();
    var range = try execute(std.testing.allocator, testNode("Range"), &.{ &start, &limit, &delta });
    defer range.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 0, 2, 4 }, range.buffer.i64);
}

test "graph dispatcher executes reduce sum and max" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 2, 2 }, &.{ 1, 4, 2, 3 });
    defer input.deinit();
    const inputs = [_]*const Tensor{&input};

    var sum = try execute(std.testing.allocator, testNode("ReduceSum"), &inputs);
    defer sum.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, sum.shape);
    try std.testing.expectEqualSlices(f32, &.{10}, sum.buffer.f32);

    var max = try execute(std.testing.allocator, testNode("ReduceMax"), &inputs);
    defer max.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, max.shape);
    try std.testing.expectEqualSlices(f32, &.{4}, max.buffer.f32);
}

test "graph dispatcher executes tile and nonzero" {
    var input = try Tensor.fromI64(std.testing.allocator, &.{ 2 }, &.{ 1, 0 });
    defer input.deinit();
    var repeats = try Tensor.fromI64(std.testing.allocator, &.{1}, &.{2});
    defer repeats.deinit();

    var tiled = try execute(std.testing.allocator, testNode("Tile"), &.{ &input, &repeats });
    defer tiled.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 0, 1, 0 }, tiled.buffer.i64);

    var nz = try execute(std.testing.allocator, testNode("NonZero"), &.{&tiled});
    defer nz.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, nz.shape);
    try std.testing.expectEqualSlices(i64, &.{ 0, 2 }, nz.buffer.i64);
}

test "graph dispatcher executes topk multi output" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 4 }, &.{ 0.1, 0.9, 0.2, 0.8 });
    defer input.deinit();
    var k = try Tensor.fromI64(std.testing.allocator, &.{1}, &.{2});
    defer k.deinit();

    var outputs = try executeAll(std.testing.allocator, testNode("TopK"), &.{ &input, &k });
    defer outputs.deinit();
    try std.testing.expectEqual(@as(usize, 2), outputs.tensors.len);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, outputs.tensors[0].shape);
    try std.testing.expectEqualSlices(f32, &.{ 0.9, 0.8 }, outputs.tensors[0].buffer.f32);
    try std.testing.expectEqualSlices(i64, &.{ 1, 3 }, outputs.tensors[1].buffer.i64);
}

test "graph dispatcher executes conservative nearest resize" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var roi = try Tensor.fromF32(std.testing.allocator, &.{0}, &.{});
    defer roi.deinit();
    var scales = try Tensor.fromF32(std.testing.allocator, &.{0}, &.{});
    defer scales.deinit();
    var sizes = try Tensor.fromI64(std.testing.allocator, &.{4}, &.{ 1, 1, 4, 4 });
    defer sizes.deinit();
    const inputs = [_]*const Tensor{ &input, &roi, &scales, &sizes };

    var output = try execute(std.testing.allocator, testNode("Resize"), &inputs);
    defer output.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 4, 4 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 }, output.buffer.f32);
}

test "graph dispatcher executes average pooling ops" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    const inputs = [_]*const Tensor{&input};
    var kernel_values = [_]i64{ 2, 2 };
    var attrs = [_]shared_graph.onnx.metadata.AttributeInfo{.{
        .allocator = std.testing.allocator,
        .name = @constCast("kernel_shape"),
        .int_values = &kernel_values,
        .int_count = kernel_values.len,
    }};
    const avg_node = shared_graph.onnx.metadata.NodeInfo{
        .allocator = std.testing.allocator,
        .name = @constCast("test"),
        .op_type = @constCast("AveragePool"),
        .domain = @constCast(""),
        .attributes = &attrs,
    };

    var avg = try execute(std.testing.allocator, avg_node, &inputs);
    defer avg.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, avg.shape);
    try std.testing.expectEqualSlices(f32, &.{2.5}, avg.buffer.f32);

    var global = try execute(std.testing.allocator, testNode("GlobalAveragePool"), &inputs);
    defer global.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, global.shape);
    try std.testing.expectEqualSlices(f32, &.{2.5}, global.buffer.f32);

    var global_max = try execute(std.testing.allocator, testNode("GlobalMaxPool"), &inputs);
    defer global_max.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, global_max.shape);
    try std.testing.expectEqualSlices(f32, &.{4}, global_max.buffer.f32);
}

test "graph dispatcher executes conv transpose" {
    var input = try Tensor.fromF32(std.testing.allocator, &.{ 1, 1, 2, 2 }, &.{ 1, 2, 3, 4 });
    defer input.deinit();
    var weights = try Tensor.fromF32(std.testing.allocator, &.{ 1, 1, 2, 2 }, &.{ 1, 1, 1, 1 });
    defer weights.deinit();

    var output = try execute(std.testing.allocator, testNode("ConvTranspose"), &.{ &input, &weights });
    defer output.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 3, 3 }, output.shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 2, 4, 10, 6, 3, 7, 4 }, output.buffer.f32);
}

fn testNode(op_type: []const u8) shared_graph.onnx.metadata.NodeInfo {
    return .{
        .allocator = std.testing.allocator,
        .name = @constCast("test"),
        .op_type = @constCast(op_type),
        .domain = @constCast(""),
    };
}
