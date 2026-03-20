const std = @import("std");
const graph = @import("graph");
const weights_mod = @import("weights");
const detect_types = @import("detect/types.zig");
const pipeline = @import("detect/pipeline.zig");

pub const Tensor = detect_types.Tensor;
pub const RuntimeError = detect_types.RuntimeError;
pub const Detection = detect_types.Detection;
pub const DetectOptions = detect_types.DetectOptions;
pub const DetectOutput = detect_types.DetectOutput;
pub const DetectProfile = detect_types.DetectProfile;
pub const DetectLevelProfile = detect_types.DetectLevelProfile;
pub const DetectBranchProfile = detect_types.DetectBranchProfile;
pub const DetectBranchKind = detect_types.DetectBranchKind;
pub const ProfiledDetectOutput = detect_types.ProfiledDetectOutput;

pub fn runDetect(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !DetectOutput {
    return pipeline.runDetect(
        output_allocator,
        tensor_allocator,
        scratch_allocator,
        model_graph,
        weights_blob,
        module_path,
        feature_inputs,
        options,
    );
}

pub fn runDetectProfile(
    output_allocator: std.mem.Allocator,
    tensor_allocator: std.mem.Allocator,
    scratch_allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    feature_inputs: []const *const Tensor,
    options: DetectOptions,
) !ProfiledDetectOutput {
    return pipeline.runDetectProfile(
        output_allocator,
        tensor_allocator,
        scratch_allocator,
        model_graph,
        weights_blob,
        module_path,
        feature_inputs,
        options,
    );
}
