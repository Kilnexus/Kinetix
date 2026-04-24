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
