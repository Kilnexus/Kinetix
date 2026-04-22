const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const spec = @import("engine_vision_base").spec;
const conv = @import("conv.zig");
const types = @import("types.zig");
const stopwatch = @import("engine_stopwatch");

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
    var timer = stopwatch.start();

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

test "SPPF add attribute applies residual input" {
    const testing = std.testing;

    const graph_json =
        \\{
        \\  "format_version": 1,
        \\  "model_name": "sppf-add-test",
        \\  "metadata": { "stride": [32.0], "class_count": 1 },
        \\  "execution_plan": [
        \\    { "index": 0, "path": "model.0", "kind": "SPPF", "from": [-1] }
        \\  ],
        \\  "module_tree": {
        \\    "path": "model.model",
        \\    "kind": "Sequential",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "model.model.0",
        \\        "kind": "SPPF",
        \\        "attrs": { "add": true },
        \\        "children": [
        \\          {
        \\            "path": "model.model.0.cv1",
        \\            "kind": "Conv",
        \\            "attrs": {
        \\              "conv2d": {
        \\                "stride": [1, 1],
        \\                "padding": [0, 0],
        \\                "groups": 1
        \\              },
        \\              "activation": "Identity"
        \\            },
        \\            "children": []
        \\          },
        \\          {
        \\            "path": "model.model.0.cv2",
        \\            "kind": "Conv",
        \\            "attrs": {
        \\              "conv2d": {
        \\                "stride": [1, 1],
        \\                "padding": [0, 0],
        \\                "groups": 1
        \\              },
        \\              "activation": "Identity"
        \\            },
        \\            "children": []
        \\          },
        \\          {
        \\            "path": "model.model.0.m",
        \\            "kind": "MaxPool2d",
        \\            "attrs": { "stride": 1, "padding": 0 },
        \\            "children": []
        \\          }
        \\        ]
        \\      }
        \\    ]
        \\  },
        \\  "tensors": [
        \\    { "name": "model.0.cv1.conv.weight", "shape": [1, 1, 1, 1], "offset": 0, "nbytes": 4 },
        \\    { "name": "model.0.cv1.conv.bias", "shape": [1], "offset": 4, "nbytes": 4 },
        \\    { "name": "model.0.cv2.conv.weight", "shape": [1, 4, 1, 1], "offset": 8, "nbytes": 16 },
        \\    { "name": "model.0.cv2.conv.bias", "shape": [1], "offset": 24, "nbytes": 4 }
        \\  ]
        \\}
    ;

    var model_graph = try graph.parseGraph(testing.allocator, graph_json);
    defer model_graph.deinit();

    var weights_data = [_]f32{
        1.0, // cv1 identity weight
        0.0, // cv1 bias
        0.0, 0.0, 0.0, 0.0, // cv2 ignores SPPF concat input
        2.0, // cv2 bias
    };
    const weights_blob = weights_mod.WeightsBlob{
        .allocator = testing.allocator,
        .data = weights_data[0..],
    };

    var input = try Tensor.init(testing.allocator, 1, 1, 2, 2);
    defer input.deinit();
    @memcpy(input.data, &[_]f32{ 1.0, 3.0, 5.0, 7.0 });

    var output = try runSPPF(testing.allocator, &model_graph, &weights_blob, "model.model.0", &input);
    defer output.deinit();

    try testing.expectEqualSlices(f32, &[_]f32{ 3.0, 5.0, 7.0, 9.0 }, output.data);
}
