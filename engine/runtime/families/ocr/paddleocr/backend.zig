const std = @import("std");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const types = @import("../../../types.zig");
const report = @import("backend/report.zig");
const inference = @import("inference/index.zig");
const postprocess = @import("postprocess/index.zig");
const resolver = @import("resolver.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .paddleocr_ocr,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    layout: resolver.Layout,
    onnx_metadata: inference.planning.onnx_metadata.Summary = .{},

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.base.model_dir);
        allocator.destroy(self);
    }
};

fn open(
    allocator: std.mem.Allocator,
    model: *const normalized.NormalizedModel,
) !?*anyopaque {
    const state = try allocator.create(State);
    errdefer allocator.destroy(state);
    state.* = .{
        .base = .{
            .provider_key = .paddleocr_ocr,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .layout = resolver.inspectLayout(model.artifacts.model_dir),
        .onnx_metadata = inference.planning.onnx_metadata.inspect(allocator, model.artifacts.model_dir) catch inference.planning.onnx_metadata.Summary{},
    };
    return state;
}

fn deinit(allocator: std.mem.Allocator, raw_state: ?*anyopaque) void {
    const raw = raw_state orelse return;
    const state: *State = @ptrCast(@alignCast(raw));
    state.destroy(allocator);
}

fn execute(
    allocator: std.mem.Allocator,
    handle: *const handle_mod.ModelHandle,
    request: types.RuntimeRequest,
) !types.RuntimeResult {
    if (request.resolvedOperationId() != .ocr) return error.OperationNotSupported;
    switch (request.input) {
        .image_path, .document_path => {},
        else => return error.InvalidInputPayload,
    }

    const state = stateFromHandle(handle);
    const layout = if (state) |loaded| loaded.layout else resolver.inspectLayout(handle.normalized.artifacts.model_dir);
    const onnx_metadata = if (state) |loaded|
        loaded.onnx_metadata
    else
        inference.planning.onnx_metadata.inspect(allocator, handle.normalized.artifacts.model_dir) catch inference.planning.onnx_metadata.Summary{};
    const image_path = switch (request.input) {
        .image_path => request.input.asString(),
        else => null,
    };
    var image_graph_model_path: ?[]u8 = null;
    defer if (image_graph_model_path) |path| allocator.free(path);
    var image_graph_result: ?inference.runtime.ImageModelResult = null;
    defer if (image_graph_result) |*result| result.deinit();
    var image_graph_error: ?[]const u8 = null;
    var postprocess_error: ?[]const u8 = null;
    var detected_boxes: ?[]postprocess.db.Box = null;
    defer if (detected_boxes) |boxes| allocator.free(boxes);
    var dictionary_path: ?[]u8 = null;
    defer if (dictionary_path) |path| allocator.free(path);
    var dictionary: ?postprocess.dictionary.Dictionary = null;
    defer if (dictionary) |*loaded| loaded.deinit();
    var recognized_text: ?postprocess.ctc.DecodedText = null;
    defer if (recognized_text) |*text| text.deinit();
    var pipeline_result: ?inference.runtime.PipelineResult = null;
    defer if (pipeline_result) |*result| result.deinit();
    var pipeline_error: ?[]const u8 = null;
    if (image_path) |path| {
        pipeline_result = inference.runtime.executePipeline(allocator, handle.normalized.artifacts.model_dir, path) catch |err| blk: {
            pipeline_error = @errorName(err);
            break :blk null;
        };
        image_graph_model_path = inference.runtime.findFirstOnnxModelFile(allocator, handle.normalized.artifacts.model_dir) catch |err| blk: {
            image_graph_error = @errorName(err);
            break :blk null;
        };
        if (image_graph_model_path) |model_path| {
            image_graph_result = inference.runtime.executeImageModelFile(allocator, model_path, path, .{}) catch |err| blk: {
                image_graph_error = @errorName(err);
                break :blk null;
            };
        }
    }
    const image_graph_stage = if (image_graph_model_path) |model_path| inference.runtime.stageFromPath(model_path) else .unknown;
    if (image_graph_result) |*result| {
        switch (image_graph_stage) {
            .det => {
                detected_boxes = inference.runtime.boxesFromDetectionResult(allocator, &result.outputs, .{}) catch |err| blk: {
                    postprocess_error = @errorName(err);
                    break :blk null;
                };
            },
            .rec => {
                dictionary_path = inference.runtime.findFirstDictionaryFile(allocator, handle.normalized.artifacts.model_dir) catch |err| blk: {
                    postprocess_error = @errorName(err);
                    break :blk null;
                };
                if (dictionary_path) |path| {
                    dictionary = postprocess.dictionary.loadFromFile(allocator, path) catch |err| blk: {
                        postprocess_error = @errorName(err);
                        break :blk null;
                    };
                }
                if (dictionary) |loaded| {
                    recognized_text = inference.runtime.decodeRecognitionResult(allocator, &result.outputs, loaded.tokens) catch |err| blk: {
                        postprocess_error = @errorName(err);
                        break :blk null;
                    };
                }
            },
            else => {},
        }
    }

    return try report.runtimeResult(allocator, .{
        .provider_key = handle.normalized.provider_key.name(),
        .model_family = handle.normalized.descriptor.family,
        .model_id = handle.normalized.descriptor.id,
        .operation = request.operation,
        .input = request.input.asString(),
        .layout = layout,
        .onnx_metadata = onnx_metadata,
        .image_graph_model_path = image_graph_model_path,
        .image_graph_stage = image_graph_stage,
        .image_graph_result = if (image_graph_result) |*result| result else null,
        .image_graph_error = image_graph_error,
        .detection_box_count = if (detected_boxes) |boxes| boxes.len else 0,
        .recognized_text = if (recognized_text) |*text| text else null,
        .dictionary_path = dictionary_path,
        .postprocess_error = postprocess_error,
        .pipeline_result = if (pipeline_result) |*result| result else null,
        .pipeline_error = pipeline_error,
    });
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
