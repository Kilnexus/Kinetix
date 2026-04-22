const std = @import("std");
const backend_mod = @import("../backend.zig");
const chandra_native = @import("../../providers/chandra_native.zig");
const chandra_preprocess = @import("../../providers/chandra_preprocess.zig");
const chandra_store = @import("../../providers/chandra_store.zig");
const chandra_weights = @import("../../providers/chandra_weights.zig");
const handle_mod = @import("../../model/handle.zig");
const normalized = @import("../../model/resolver/normalized_model.zig");
const types = @import("../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .chandra_ocr,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    readiness: chandra_native.Readiness,
    parsed_config: ?chandra_native.ParsedConfig = null,
    image_processor: ?chandra_preprocess.ParsedImageProcessorConfig = null,
    manifest: ?chandra_weights.TensorManifest = null,
    tensor_store: ?chandra_store.ChandraStore = null,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        if (self.tensor_store) |*tensor_store| tensor_store.deinit();
        if (self.manifest) |*manifest| manifest.deinit();
        if (self.image_processor) |*image_processor| image_processor.deinit();
        if (self.parsed_config) |*parsed_config| parsed_config.deinit();
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
            .provider_key = .chandra_ocr,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .readiness = chandra_native.inspect(model.artifacts.model_dir),
    };
    errdefer allocator.free(state.base.model_dir);

    if (model.artifacts.config_path) |config_path| {
        state.parsed_config = chandra_native.loadConfigFromFile(allocator, config_path) catch null;
    }
    errdefer if (state.parsed_config) |*parsed_config| parsed_config.deinit();

    state.image_processor = chandra_preprocess.loadImageProcessorConfig(allocator, model.artifacts.model_dir) catch null;
    errdefer if (state.image_processor) |*image_processor| image_processor.deinit();

    state.manifest = chandra_weights.loadManifest(allocator, model.artifacts.model_dir) catch null;
    errdefer if (state.manifest) |*manifest| manifest.deinit();

    state.tensor_store = chandra_store.ChandraStore.open(allocator, model.artifacts.model_dir) catch null;
    errdefer if (state.tensor_store) |*tensor_store| tensor_store.deinit();

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
    switch (request.input) {
        .image_path, .document_path => {},
        else => return error.InvalidInputPayload,
    }
    const input_path = request.input.asString() orelse return error.MissingInputPayload;
    const context = chandra_native.Context{
        .operation = request.operation,
        .model_path = handle.normalized.artifacts.model_dir,
        .input_path = input_path,
        .execution = request.execution,
        .max_output_tokens = request.generation.max_tokens,
    };
    const maybe_state = stateFromHandle(handle);
    const output = if (maybe_state) |state|
        try chandra_native.executeWithLoadedModel(allocator, context, .{
            .readiness = state.readiness,
            .parsed_config = if (state.parsed_config) |*config| config else null,
            .image_processor = if (state.image_processor) |*image_processor| image_processor else null,
            .tensor_store = if (state.tensor_store) |*tensor_store| tensor_store else null,
        })
    else
        try chandra_native.execute(allocator, context);
    return .{
        .origin = .shared_adapter,
        .note = .ocr_chandra_native,
        .output = .{ .json = output },
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
