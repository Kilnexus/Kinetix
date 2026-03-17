const graph = @import("graph");
const weights_mod = @import("weights");
const execute = @import("execute.zig");
const psa = @import("psa.zig");
const spec = @import("spec.zig");
const Tensor = @import("types.zig").Tensor;
const Activation = @import("types.zig").Activation;

test "weightPrefixForModulePath normalizes exported module paths" {
    const testing = @import("std").testing;

    var buffer: [128]u8 = undefined;
    try testing.expectEqualStrings(
        "model.2.cv1",
        try spec.weightPrefixForModulePath(&buffer, "model.model.2.cv1"),
    );
    try testing.expectEqualStrings(
        "model.23.cv2.0.2",
        try spec.weightPrefixForModulePath(&buffer, "model.model.23.cv2.0.2"),
    );
}

test "resolveConvSpec reads wrapped and bare conv metadata from exported graph" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();

    const wrapped = try spec.resolveConvSpec(&model_graph, "model.model.2.cv1");
    try testing.expectEqualStrings("model.2.cv1.conv.weight", wrapped.weight.name);
    try testing.expectEqualStrings("model.2.cv1.conv.bias", wrapped.bias.?.name);
    try testing.expectEqual(Activation.silu, wrapped.activation);
    try testing.expectEqual(@as(usize, 1), wrapped.stride[0]);
    try testing.expectEqual(@as(usize, 0), wrapped.padding[0]);

    const bare = try spec.resolveConvSpec(&model_graph, "model.model.23.cv2.0.2");
    try testing.expectEqualStrings("model.23.cv2.0.2.weight", bare.weight.name);
    try testing.expectEqualStrings("model.23.cv2.0.2.bias", bare.bias.?.name);
    try testing.expectEqual(Activation.identity, bare.activation);
    try testing.expectEqual(@as(usize, 1), bare.stride[0]);
    try testing.expectEqual(@as(usize, 0), bare.padding[0]);
}

test "runConvModule executes a wrapped conv from exported weights" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 64, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runConvModule(testing.allocator, &model_graph, &weights_blob, "model.model.2.cv1", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 64, 8, 8 }, &output.shape);
}

test "runBottleneck preserves tensor shape for residual block" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 32, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runBottleneck(testing.allocator, &model_graph, &weights_blob, "model.model.2.m.0", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &input.shape, &output.shape);
}

test "runSPPF executes pooled projection block" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 512, 4, 4);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runSPPF(testing.allocator, &model_graph, &weights_blob, "model.model.9", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 512, 4, 4 }, &output.shape);
}

test "runC3k executes composite branch block from exported weights" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 128, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runC3k(testing.allocator, &model_graph, &weights_blob, "model.model.6.m.0", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 128, 8, 8 }, &output.shape);
}

test "runC3k2 executes variant with nested C3k child" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 256, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runC3k2(testing.allocator, &model_graph, &weights_blob, "model.model.6", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 256, 8, 8 }, &output.shape);
}

test "runC3k2 executes variant with bottleneck child" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 768, 8, 8);
    defer input.deinit();
    input.fill(0.0);

    var output = try execute.runC3k2(testing.allocator, &model_graph, &weights_blob, "model.model.13", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 256, 8, 8 }, &output.shape);
}

test "runAttention executes exported PSA attention block" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 256, 10, 10);
    defer input.deinit();
    input.fill(0.0);

    var output = try psa.runAttention(testing.allocator, &model_graph, &weights_blob, "model.model.10.m.0.attn", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 256, 10, 10 }, &output.shape);
}

test "runPSABlock executes attention plus ffn residual block" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 256, 10, 10);
    defer input.deinit();
    input.fill(0.0);

    var output = try psa.runPSABlock(testing.allocator, &model_graph, &weights_blob, "model.model.10.m.0", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 256, 10, 10 }, &output.shape);
}

test "runC2PSA executes exported layer 10 block" {
    const testing = @import("std").testing;

    var model_graph = try graph.load(testing.allocator, "artifacts/graph.json");
    defer model_graph.deinit();
    var weights_blob = try weights_mod.WeightsBlob.load(testing.allocator, "artifacts/weights.bin");
    defer weights_blob.deinit();

    var input = try Tensor.init(testing.allocator, 1, 512, 10, 10);
    defer input.deinit();
    input.fill(0.0);

    var output = try psa.runC2PSA(testing.allocator, &model_graph, &weights_blob, "model.model.10", &input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, &[_]usize{ 1, 512, 10, 10 }, &output.shape);
}
