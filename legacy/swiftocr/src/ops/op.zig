pub const OpType = enum {
    input,
    conv2d,
    depthwise_conv2d,
    matmul,
    batch_norm,
    relu,
    hard_swish,
    max_pool2d,
    avg_pool2d,
    concat,
    reshape,
    softmax,
};
