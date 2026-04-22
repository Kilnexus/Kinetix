const core = @import("model/core.zig");
const input = @import("input/loader.zig");
const execute_mod = @import("execute/runner.zig");

pub const io = core.io;

pub const TextConfig = core.TextConfig;
pub const VisionConfig = core.VisionConfig;
pub const Config = core.Config;
pub const ParsedConfig = core.ParsedConfig;
pub const Context = core.Context;
pub const Readiness = core.Readiness;
pub const LoadedModel = core.LoadedModel;
pub const PreprocessSummary = core.PreprocessSummary;
pub const MultimodalPositionPlan = core.MultimodalPositionPlan;

pub const visualTokenPosition = core.visualTokenPosition;
pub const textTokenPosition = core.textTokenPosition;
pub const maxVisualPosition = core.maxVisualPosition;
pub const allocMultimodalTextPositions = core.allocMultimodalTextPositions;
pub const allocMultimodalVisualPositions = core.allocMultimodalVisualPositions;
pub const inspect = core.inspect;
pub const loadConfigFromFile = core.loadConfigFromFile;

pub const isSupportedInputPath = input.isSupportedInputPath;
pub const isRasterImagePath = input.isRasterImagePath;
pub const isFrameManifestPath = input.isFrameManifestPath;
pub const isDirectoryPath = input.isDirectoryPath;
pub const loadPreparedInputFromPath = input.loadPreparedInputFromPath;
pub const loadPreparedInputFromDirectory = input.loadPreparedInputFromDirectory;
pub const loadPreparedInputFromManifest = input.loadPreparedInputFromManifest;

pub const execute = execute_mod.execute;
pub const executeWithLoadedModel = execute_mod.executeWithLoadedModel;

test {
    _ = @import("testing/native_test.zig");
}
