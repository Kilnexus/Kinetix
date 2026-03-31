pub const core = struct {
    pub const Tensor = @import("core/tensor.zig").Tensor;
    pub const DataType = @import("core/tensor.zig").DataType;
    pub const MemoryPool = @import("core/memory_pool.zig").MemoryPool;
    pub const ThreadPool = @import("core/thread_pool.zig").ThreadPool;
};

pub const graph = struct {
    pub const Graph = @import("graph/graph.zig").Graph;
    pub const Node = @import("graph/graph.zig").Node;
};

pub const ops = struct {
    pub const OpType = @import("ops/op.zig").OpType;
    pub const relu = @import("ops/activation.zig").relu;
    pub const matmul = @import("ops/matmul.zig").matmul;
};

pub const io = struct {
    pub const Model = @import("io/model.zig").Model;
    pub const Image = @import("io/image.zig").Image;
};

pub const ocr = struct {
    pub const Pipeline = @import("ocr/pipeline.zig").Pipeline;
};

test {
    _ = @import("core/tensor.zig");
    _ = @import("graph/graph.zig");
    _ = @import("ops/activation.zig");
    _ = @import("ops/matmul.zig");
    _ = @import("io/model.zig");
    _ = @import("io/image.zig");
    _ = @import("ocr/pipeline.zig");
}
