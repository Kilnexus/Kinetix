const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const spec = @import("engine_vision_base").spec;
const conv = @import("conv.zig");
const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const SPPFProfiledTensor = types.SPPFProfiledTensor;

pub fn runSPPF(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runSPPFNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runSPPFNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !Tensor {
    if (!std.mem.eql(u8, module.kind, "SPPF")) return error.InvalidModuleKind;

    const pool = &module.children[2];
    const stride = try spec.getNodePair(pool, "stride");
    const padding = try spec.getNodePair(pool, "padding");
    const kernel = .{ padding[0] * 2 + 1, padding[1] * 2 + 1 };
    const has_add = module.cached_attrs.add orelse false;

    var base = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    defer base.deinit();

    var pool1 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool1.deinit();
    try ops.maxPool2d(&base, &pool1, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var pool2 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool2.deinit();
    try ops.maxPool2d(&pool1, &pool2, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var pool3 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool3.deinit();
    try ops.maxPool2d(&pool2, &pool3, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);

    var concat = try Tensor.init(
        allocator,
        base.shape[0],
        base.shape[1] * 4,
        base.shape[2],
        base.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &base, &pool1, &pool2, &pool3 };
    try ops.concatChannels(&inputs, &concat);

    var output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    errdefer output.deinit();
    if (has_add) try ops.addInPlace(&output, input);
    return output;
}

pub fn runSPPFProfile(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !SPPFProfiledTensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    return runSPPFProfileNode(allocator, model_graph, weights_blob, module, input);
}

pub fn runSPPFProfileNode(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module: *const graph.ModuleNode,
    input: *const Tensor,
) !SPPFProfiledTensor {
    if (!std.mem.eql(u8, module.kind, "SPPF")) return error.InvalidModuleKind;

    const pool = &module.children[2];
    const stride = try spec.getNodePair(pool, "stride");
    const padding = try spec.getNodePair(pool, "padding");
    const kernel = .{ padding[0] * 2 + 1, padding[1] * 2 + 1 };
    const has_add = module.cached_attrs.add orelse false;

    var profile = types.SPPFProfile{};
    var timer = try std.time.Timer.start();

    var base = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[0], input);
    profile.cv1_ns = timer.read();
    defer base.deinit();

    timer.reset();
    var pool1 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool1.deinit();
    try ops.maxPool2d(&base, &pool1, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool1_ns = timer.read();

    timer.reset();
    var pool2 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool2.deinit();
    try ops.maxPool2d(&pool1, &pool2, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool2_ns = timer.read();

    timer.reset();
    var pool3 = try Tensor.init(allocator, base.shape[0], base.shape[1], base.shape[2], base.shape[3]);
    defer pool3.deinit();
    try ops.maxPool2d(&pool2, &pool3, kernel[0], kernel[1], stride[0], stride[1], padding[0], padding[1]);
    profile.pool3_ns = timer.read();

    timer.reset();
    var concat = try Tensor.init(
        allocator,
        base.shape[0],
        base.shape[1] * 4,
        base.shape[2],
        base.shape[3],
    );
    defer concat.deinit();

    const inputs = [_]*const Tensor{ &base, &pool1, &pool2, &pool3 };
    try ops.concatChannels(&inputs, &concat);
    profile.concat_ns = timer.read();

    timer.reset();
    var output = try conv.runConvNode(allocator, model_graph, weights_blob, &module.children[1], &concat);
    errdefer output.deinit();
    if (has_add) try ops.addInPlace(&output, input);
    profile.cv2_ns = timer.read();
    return .{ .output = output, .sppf_profile = profile };
}
