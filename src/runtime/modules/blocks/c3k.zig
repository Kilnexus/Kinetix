const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const bottleneck = @import("bottleneck.zig");
const conv = @import("conv.zig");
const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const C3kProfiledTensor = types.C3kProfiledTensor;

pub fn runC3kNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
    module_runner: types.ModuleRunnerFn,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "C3k")) return error.InvalidModuleKind;

    var left = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer left.deinit();

    const seq_node = &module.children[3];
    if (seq_node.children.len == 2 and
        std.mem.eql(u8, seq_node.children[0].kind, "Bottleneck") and
        std.mem.eql(u8, seq_node.children[1].kind, "Bottleneck"))
    {
        const next0 = try bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[0], &left);
        left.deinit();
        left = next0;

        const next1 = try bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[1], &left);
        left.deinit();
        left = next1;
    } else {
        for (seq_node.children) |child| {
            const next = try module_runner(allocator, model_graph, weights_blob, &child, &left);
            left.deinit();
            left = next;
        }
    }

    var right = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], input);
    defer right.deinit();

    var concat = try Tensor.init(
        allocator,
        left.shape[0],
        left.shape[1] + right.shape[1],
        left.shape[2],
        left.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &left, &right };
    try ops.concatChannels(&inputs, &concat);

    return try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[2], &concat);
}

pub fn runC3kProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
    module_runner: types.ModuleRunnerFn,
) !C3kProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "C3k")) return error.InvalidModuleKind;

    var profile = types.C3kProfile{};
    var timer = try std.time.Timer.start();

    var left = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer left.deinit();

    const seq_node = &module.children[3];
    if (seq_node.children.len == 2 and
        std.mem.eql(u8, seq_node.children[0].kind, "Bottleneck") and
        std.mem.eql(u8, seq_node.children[1].kind, "Bottleneck"))
    {
        profile.seq_kind = "Bottleneckx2";
        timer.reset();
        const next0 = try bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[0], &left);
        left.deinit();
        left = next0;

        const next1 = try bottleneck.runBottleneckNode(allocator, model_graph, weights_blob, &seq_node.children[1], &left);
        left.deinit();
        left = next1;
        profile.seq_ns = timer.read();
    } else {
        profile.seq_kind = seq_node.kind;
        timer.reset();
        for (seq_node.children) |child| {
            const next = try module_runner(allocator, model_graph, weights_blob, &child, &left);
            left.deinit();
            left = next;
        }
        profile.seq_ns = timer.read();
    }

    timer.reset();
    var right = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], input);
    profile.cv2_ns = timer.read();
    defer right.deinit();

    timer.reset();
    var concat = try Tensor.init(
        allocator,
        left.shape[0],
        left.shape[1] + right.shape[1],
        left.shape[2],
        left.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &left, &right };
    try ops.concatChannels(&inputs, &concat);
    profile.concat_ns = timer.read();

    timer.reset();
    const output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[2], &concat);
    profile.cv3_ns = timer.read();
    return .{ .output = output, .c3k_profile = profile };
}
