const native = @import("chandra/native/index.zig");

pub const io = native.io;

pub const TextConfig = native.TextConfig;
pub const VisionConfig = native.VisionConfig;
pub const Config = native.Config;
pub const ParsedConfig = native.ParsedConfig;
pub const Context = native.Context;
pub const Readiness = native.Readiness;
pub const LoadedModel = native.LoadedModel;
pub const PreprocessSummary = native.PreprocessSummary;
pub const MultimodalPositionPlan = native.MultimodalPositionPlan;

pub const visualTokenPosition = native.visualTokenPosition;
pub const textTokenPosition = native.textTokenPosition;
pub const maxVisualPosition = native.maxVisualPosition;
pub const allocMultimodalTextPositions = native.allocMultimodalTextPositions;
pub const allocMultimodalVisualPositions = native.allocMultimodalVisualPositions;
pub const inspect = native.inspect;
pub const loadConfigFromFile = native.loadConfigFromFile;

pub const isSupportedInputPath = native.isSupportedInputPath;
pub const isRasterImagePath = native.isRasterImagePath;
pub const isFrameManifestPath = native.isFrameManifestPath;
pub const isDirectoryPath = native.isDirectoryPath;
pub const loadPreparedInputFromPath = native.loadPreparedInputFromPath;
pub const loadPreparedInputFromDirectory = native.loadPreparedInputFromDirectory;
pub const loadPreparedInputFromManifest = native.loadPreparedInputFromManifest;

pub const execute = native.execute;
pub const executeWithLoadedModel = native.executeWithLoadedModel;

test {
    _ = @import("chandra/native/testing/native_test.zig");
}
