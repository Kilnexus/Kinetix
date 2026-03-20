const common = @import("common.zig");
const stride1 = @import("kernel_3x3_stride1.zig");
const stride2 = @import("kernel_3x3_stride2.zig");

pub fn conv2d3x3Pad1(
    input: *const common.Tensor,
    weights: *const common.Tensor,
    bias: ?[]const f32,
    output: *common.Tensor,
    options: common.Conv2DOptions,
) common.OpError!void {
    if (options.stride_h == 2 and options.stride_w == 2) {
        const batch = input.shape[0];
        const out_channels = weights.shape[0];
        const expected_h = output.shape[2];
        const expected_w = output.shape[3];
        const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
        const thread_count = common.chooseConvThreadCount(workload, out_channels);
        if (thread_count > 1) {
            return stride2.conv2d3x3Pad1Stride2Parallel(input, weights, bias, output, options, thread_count);
        }
        return stride2.conv2d3x3Pad1Stride2Range(input, weights, bias, output, options, 0, out_channels);
    }

    const batch = input.shape[0];
    const out_channels = weights.shape[0];
    const expected_h = output.shape[2];
    const expected_w = output.shape[3];
    const workload = batch * out_channels * expected_h * expected_w * input.shape[1] * 9;
    const thread_count = common.chooseConvThreadCount(workload, out_channels);
    if (thread_count > 1) {
        return stride1.conv2d3x3Pad1Parallel(input, weights, bias, output, options, thread_count);
    }
    return stride1.conv2d3x3Pad1Range(input, weights, bias, output, options, 0, out_channels);
}
