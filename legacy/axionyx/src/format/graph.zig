const std = @import("std");
const shared = @import("engine_vision_graph");

pub const AttrEntry = shared.AttrEntry;
pub const AttrValue = shared.AttrValue;
pub const ModuleNode = shared.ModuleNode;
pub const TensorMeta = shared.TensorMeta;
pub const ExecutionNode = shared.ExecutionNode;
pub const Graph = shared.Graph;
pub const Summary = shared.Summary;

pub const load = shared.load;
pub const loadSummary = shared.loadSummary;
pub const parseGraph = shared.parseGraph;

test "parseGraph exposes module tree and attrs" {
    const testing = std.testing;

    const raw =
        \\{
        \\  "format_version": 1,
        \\  "model_name": "mini",
        \\  "metadata": { "class_count": 2, "stride": [8.0, 16.0] },
        \\  "tensors": [],
        \\  "execution_plan": [
        \\    { "index": 0, "path": "model.0", "kind": "Conv", "from": [-1] }
        \\  ],
        \\  "module_tree": {
        \\    "path": "model",
        \\    "kind": "Root",
        \\    "attrs": {},
        \\    "children": [
        \\      {
        \\        "path": "model.0",
        \\        "kind": "Conv",
        \\        "attrs": {
        \\          "activation": "SiLU",
        \\          "conv2d": {
        \\            "out_channels": 16,
        \\            "kernel_size": [3, 3],
        \\            "stride": [2, 2],
        \\            "padding": [1, 1],
        \\            "groups": 1
        \\          }
        \\        },
        \\        "children": []
        \\      }
        \\    ]
        \\  }
        \\}
    ;

    var model_graph = try parseGraph(testing.allocator, raw);
    defer model_graph.deinit();

    try testing.expectEqual(@as(i64, 1), model_graph.format_version);
    try testing.expectEqualStrings("mini", model_graph.model_name);
    try testing.expectEqual(@as(i64, 2), model_graph.class_count);
    try testing.expectEqual(@as(usize, 2), model_graph.strides.len);
    try testing.expectEqual(@as(usize, 1), model_graph.execution_nodes.len);

    const module = model_graph.findModule("model.0") orelse return error.ModuleNotFound;
    try testing.expectEqualStrings("Conv", module.kind);
    try testing.expect(module.getAttr("activation") != null);
    try testing.expect(module.cached_conv.valid);
    try testing.expectEqual(@as(usize, 2), module.cached_conv.stride[0]);
    try testing.expectEqual(@as(usize, 1), module.cached_conv.padding[0]);
    try testing.expectEqual(@as(usize, 1), module.cached_conv.groups);
    try testing.expect(module.cached_conv.apply_silu);
}
