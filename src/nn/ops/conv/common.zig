const types = @import("../types.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
pub const Conv2DOptions = types.Conv2DOptions;

pub const max_supported_conv_threads = 4;
pub const conv_parallel_min_workload = 2_000_000;
pub const conv_parallel_two_thread_workload = 2_500_000;
pub const simd_lane_count = 8;
pub const F32xN = @Vector(simd_lane_count, f32);

pub inline fn siluValue(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub inline fn maybeApplySilu(x: f32, apply_silu: bool) f32 {
    return if (apply_silu) siluValue(x) else x;
}

pub inline fn loadF32xN(slice: []const f32, index: usize) F32xN {
    return slice[index..][0..simd_lane_count].*;
}

pub inline fn storeF32xN(slice: []f32, index: usize, value: F32xN) void {
    slice[index..][0..simd_lane_count].* = value;
}

pub fn chooseConvThreadCount(workload: usize, out_channels: usize) usize {
    if (out_channels < 2 or workload < conv_parallel_min_workload) return 1;
    if (workload < conv_parallel_two_thread_workload) return @min(out_channels, 2);
    return @min(out_channels, max_supported_conv_threads);
}
