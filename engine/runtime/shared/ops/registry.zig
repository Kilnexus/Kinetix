const std = @import("std");

pub const Domain = enum {
    onnx_graph,
    vision_nn,
    text_core,
    text_attention,
    text_gqa,
    text_quantized,
};

pub const Status = enum {
    graph_executable,
    native_kernel,
    adapter_required,
};

pub const Entry = struct {
    name: []const u8,
    domain: Domain,
    status: Status,
    module: []const u8,
};

pub const Summary = struct {
    total: usize = 0,
    onnx_graph: usize = 0,
    vision_nn: usize = 0,
    text_core: usize = 0,
    text_attention: usize = 0,
    text_gqa: usize = 0,
    text_quantized: usize = 0,
    graph_executable: usize = 0,
    native_kernel: usize = 0,
    adapter_required: usize = 0,
};

pub const entries = [_]Entry{
    .{ .name = "Constant", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "Identity", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "Add", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "Mul", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "Relu", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "Sigmoid", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/activation/index.zig" },
    .{ .name = "LeakyRelu", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/activation/index.zig" },
    .{ .name = "Gelu", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/activation/index.zig" },
    .{ .name = "SwiGLU", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/activation/index.zig" },
    .{ .name = "Cast", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/core/index.zig" },
    .{ .name = "MatMul", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/linear/index.zig" },
    .{ .name = "Gemm", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/linear/index.zig" },
    .{ .name = "Reshape", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Shape", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Unsqueeze", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Squeeze", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Concat", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Transpose", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/shape/index.zig" },
    .{ .name = "Gather", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/indexing/index.zig" },
    .{ .name = "Slice", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/indexing/index.zig" },
    .{ .name = "Softmax", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/normalization/index.zig" },
    .{ .name = "LayerNormalization", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/normalization/index.zig" },
    .{ .name = "RMSNormalization", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/normalization/index.zig" },
    .{ .name = "Conv", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/spatial/index.zig" },
    .{ .name = "MaxPool", .domain = .onnx_graph, .status = .graph_executable, .module = "shared/ops/graph/spatial/index.zig" },

    .{ .name = "SiLU", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },
    .{ .name = "Sigmoid", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },
    .{ .name = "Add", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },
    .{ .name = "UpsampleNearest", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/layout/index.zig" },
    .{ .name = "ConcatChannels", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/layout/index.zig" },
    .{ .name = "CopyChannelRange", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/layout/index.zig" },
    .{ .name = "CopyTensorBlock", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/layout/index.zig" },
    .{ .name = "MaxPool2d", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/pooling/index.zig" },
    .{ .name = "MatMul", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/linalg/index.zig" },
    .{ .name = "SoftmaxRows", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/linalg/index.zig" },
    .{ .name = "Conv2d", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/conv/entry.zig" },
    .{ .name = "Conv2dGeneral", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/conv/general.zig" },
    .{ .name = "Conv2d3x3Pad1", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/conv/kernel_3x3.zig" },
    .{ .name = "Conv2dPointwise", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/conv/pointwise.zig" },
    .{ .name = "Conv2dPointwiseConcat", .domain = .vision_nn, .status = .native_kernel, .module = "shared/ops/kernels/conv/pointwise.zig" },

    .{ .name = "Dot", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/linalg/index.zig" },
    .{ .name = "AxpyInPlace", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/linalg/index.zig" },
    .{ .name = "MatMulVec", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/linalg/index.zig" },
    .{ .name = "RmsNorm", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/normalization/index.zig" },
    .{ .name = "LayerNorm", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/normalization/index.zig" },
    .{ .name = "RmsNormRepeated", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/normalization/index.zig" },
    .{ .name = "SiLU", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },
    .{ .name = "GELU", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },
    .{ .name = "SwiGLU", .domain = .text_core, .status = .native_kernel, .module = "shared/ops/kernels/activation/index.zig" },

    .{ .name = "SoftmaxInPlace", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/basic.zig" },
    .{ .name = "ScaledDotProductAttentionSingleQuery", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/basic.zig" },
    .{ .name = "ScaledDotProductAttentionSingleQueryBf16Cache", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/basic.zig" },
    .{ .name = "RoPE", .domain = .text_attention, .status = .native_kernel, .module = "text/attention/rope.zig" },
    .{ .name = "ScaledDotProductAttentionSingleQueryQ8Cache", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/q8/index.zig" },
    .{ .name = "ScaledDotProductAttentionSingleQueryQ8CacheHeadMajor", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/q8/index.zig" },
    .{ .name = "ScaledDotProductAttentionSingleQueryQ8CachePagedHeadMajor", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/q8/index.zig" },
    .{ .name = "DotQ8GroupedSlice", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/q8/index.zig" },
    .{ .name = "AxpyQ8GroupedSliceInPlace", .domain = .text_attention, .status = .native_kernel, .module = "shared/ops/kernels/attention/q8/index.zig" },

    .{ .name = "ApplyRoPEToProjectedHeads", .domain = .text_gqa, .status = .native_kernel, .module = "text/gqa_attention/forward.zig" },
    .{ .name = "ForwardProjectedSingleToken", .domain = .text_gqa, .status = .native_kernel, .module = "shared/ops/kernels/attention/gqa/index.zig" },
    .{ .name = "ForwardProjectedSingleTokenBf16Cache", .domain = .text_gqa, .status = .native_kernel, .module = "shared/ops/kernels/attention/gqa/index.zig" },
    .{ .name = "ForwardProjectedSingleTokenQ8Cache", .domain = .text_gqa, .status = .native_kernel, .module = "shared/ops/kernels/attention/gqa/index.zig" },
    .{ .name = "ForwardProjectedSingleTokenQ8CacheHeadMajor", .domain = .text_gqa, .status = .native_kernel, .module = "shared/ops/kernels/attention/gqa/index.zig" },
    .{ .name = "ForwardProjectedSingleTokenQ8CachePagedHeadMajor", .domain = .text_gqa, .status = .native_kernel, .module = "shared/ops/kernels/attention/gqa/index.zig" },

    .{ .name = "EncodeQ8Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "EncodeQ6Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "EncodeQ4Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DecodeQ8Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DecodeQ6Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DecodeQ4Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DotQ8Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DotQ6Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "DotQ4Row", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "MatMulQ8Rows", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "MatMulQ6Rows", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },
    .{ .name = "MatMulQ4Rows", .domain = .text_quantized, .status = .native_kernel, .module = "shared/ops/kernels/quantized/index.zig" },

};

pub fn all() []const Entry {
    return entries[0..];
}

pub fn find(name: []const u8, domain: Domain) ?Entry {
    for (entries) |entry| {
        if (entry.domain == domain and std.mem.eql(u8, entry.name, name)) return entry;
    }
    return null;
}

pub fn has(name: []const u8, domain: Domain) bool {
    return find(name, domain) != null;
}

pub fn isGraphExecutableOnnx(name: []const u8) bool {
    if (find(name, .onnx_graph)) |entry| return entry.status == .graph_executable;
    return false;
}

pub fn hasNativeKernel(name: []const u8) bool {
    for (entries) |entry| {
        if (entry.status == .native_kernel and std.mem.eql(u8, entry.name, name)) return true;
    }
    return false;
}

pub fn countByDomain(domain: Domain) usize {
    var count: usize = 0;
    for (entries) |entry| {
        if (entry.domain == domain) count += 1;
    }
    return count;
}

pub fn countByStatus(status: Status) usize {
    var count: usize = 0;
    for (entries) |entry| {
        if (entry.status == status) count += 1;
    }
    return count;
}

pub fn summarize() Summary {
    var summary = Summary{ .total = entries.len };
    for (entries) |entry| {
        switch (entry.domain) {
            .onnx_graph => summary.onnx_graph += 1,
            .vision_nn => summary.vision_nn += 1,
            .text_core => summary.text_core += 1,
            .text_attention => summary.text_attention += 1,
            .text_gqa => summary.text_gqa += 1,
            .text_quantized => summary.text_quantized += 1,
        }
        switch (entry.status) {
            .graph_executable => summary.graph_executable += 1,
            .native_kernel => summary.native_kernel += 1,
            .adapter_required => summary.adapter_required += 1,
        }
    }
    return summary;
}

test "unified ops registry includes graph executable and native kernels" {
    try std.testing.expect(isGraphExecutableOnnx("MatMul"));
    try std.testing.expect(isGraphExecutableOnnx("LayerNormalization"));
    try std.testing.expect(isGraphExecutableOnnx("RMSNormalization"));
    try std.testing.expect(isGraphExecutableOnnx("Sigmoid"));
    try std.testing.expect(isGraphExecutableOnnx("Conv"));
    try std.testing.expect(isGraphExecutableOnnx("MaxPool"));
    try std.testing.expect(isGraphExecutableOnnx("SwiGLU"));
    try std.testing.expect(has("Conv2d", .vision_nn));
    try std.testing.expect(has("RmsNorm", .text_core));
    try std.testing.expect(has("MatMulQ8Rows", .text_quantized));
    try std.testing.expect(countByStatus(.graph_executable) >= 18);
    try std.testing.expect(countByStatus(.native_kernel) >= 45);
    const summary = summarize();
    try std.testing.expectEqual(entries.len, summary.total);
    try std.testing.expectEqual(@as(usize, 0), summary.adapter_required);
}
