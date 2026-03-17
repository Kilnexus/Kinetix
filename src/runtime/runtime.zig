const std = @import("std");
const types = @import("base/types.zig");
const spec = @import("base/spec.zig");
const blocks = @import("modules/blocks.zig");
const psa = @import("modules/psa.zig");
const detect = @import("modules/detect.zig");
const graph_exec = @import("engine/graph_exec.zig");
const trace = @import("engine/trace.zig");
const inspect = @import("model/inspect.zig");

pub const TensorDesc = types.TensorDesc;
pub const Tensor = types.Tensor;
pub const Activation = types.Activation;
pub const RuntimeError = types.RuntimeError;
pub const ConvSpec = types.ConvSpec;
pub const shapeLen = types.shapeLen;

pub const resolveConvSpec = spec.resolveConvSpec;
pub const weightPrefixForModulePath = spec.weightPrefixForModulePath;
pub const getNodePair = spec.getNodePair;

pub const runConvModule = blocks.runConvModule;
pub const runBottleneck = blocks.runBottleneck;
pub const runSPPF = blocks.runSPPF;
pub const runC3k = blocks.runC3k;
pub const runC3k2 = blocks.runC3k2;
pub const runModule = blocks.runModule;
pub const runAttention = psa.runAttention;
pub const runPSABlock = psa.runPSABlock;
pub const runC2PSA = psa.runC2PSA;
pub const Detection = detect.Detection;
pub const DetectOptions = detect.DetectOptions;
pub const DetectOutput = detect.DetectOutput;
pub const runDetect = detect.runDetect;
pub const runUpsampleModule = graph_exec.runUpsampleModule;
pub const runGraph = graph_exec.runGraph;
pub const NodeTrace = trace.NodeTrace;
pub const GraphTrace = trace.GraphTrace;
pub const traceGraph = trace.traceGraph;
pub const KindCount = inspect.KindCount;
pub const SupportReport = inspect.SupportReport;
pub const inspectModel = inspect.inspectModel;
pub const isSupportedExecutionKind = inspect.isSupportedExecutionKind;
pub const isSupportedModuleKind = inspect.isSupportedModuleKind;

pub fn printRoadmap(writer: anytype) !void {
    try writer.writeAll(
        \\Full runtime status:
        \\1. Graph and weights export: ready
        \\2. Zig graph loader: ready
        \\3. Primitive tensor ops: implemented
        \\4. Module-tree spec resolution: implemented
        \\5. Composite YOLO11s blocks: implemented
        \\6. Detect + DFL + NMS: implemented
        \\7. End-to-end graph execution: implemented
        \\8. Numerical parity check: verified
        \\
    );
}
