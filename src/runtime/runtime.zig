const std = @import("std");
const types = @import("types.zig");
const spec = @import("spec.zig");
const execute = @import("execute.zig");

pub const TensorDesc = types.TensorDesc;
pub const Tensor = types.Tensor;
pub const Activation = types.Activation;
pub const RuntimeError = types.RuntimeError;
pub const ConvSpec = types.ConvSpec;
pub const shapeLen = types.shapeLen;

pub const resolveConvSpec = spec.resolveConvSpec;
pub const weightPrefixForModulePath = spec.weightPrefixForModulePath;
pub const getNodePair = spec.getNodePair;

pub const runConvModule = execute.runConvModule;
pub const runBottleneck = execute.runBottleneck;
pub const runSPPF = execute.runSPPF;
pub const runC3k = execute.runC3k;
pub const runC3k2 = execute.runC3k2;
pub const runModule = execute.runModule;

pub fn printRoadmap(writer: anytype) !void {
    try writer.writeAll(
        \\Full runtime status:
        \\1. Graph and weights export: ready
        \\2. Zig graph loader: ready
        \\3. Primitive tensor ops: implemented
        \\4. Module-tree spec resolution: implemented
        \\5. Composite YOLO11s blocks: pending
        \\6. Detect + DFL + NMS: pending
        \\7. End-to-end parity check: pending
        \\
    );
}

test {
    _ = @import("runtime_tests.zig");
}
