const std = @import("std");
const ocr_artifacts = @import("../../../../artifacts/ocr/ocr.zig");
const backend_mod = @import("../../../backend/backend.zig");
const handle_mod = @import("../../../model/handle.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const swiftocr_native = @import("native.zig");
const types = @import("../../../types.zig");

pub const backend = backend_mod.RuntimeBackend{
    .provider_key = .swiftocr_ocr,
    .open_fn = open,
    .deinit_fn = deinit,
    .execute_fn = execute,
};

const State = struct {
    base: backend_mod.OpenState,
    model: ?ocr_artifacts.Model = null,

    fn destroy(self: *State, allocator: std.mem.Allocator) void {
        if (self.model) |*model| model.deinit();
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
            .provider_key = .swiftocr_ocr,
            .model_dir = try allocator.dupe(u8, model.artifacts.model_dir),
        },
        .model = null,
    };
    errdefer allocator.free(state.base.model_dir);

    if (model.artifacts.ocr_model_path) |ocr_model_path| {
        state.model = ocr_artifacts.Model.loadFromFile(allocator, ocr_model_path) catch null;
    }
    errdefer if (state.model) |*loaded| loaded.deinit();

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
        .image_path => {},
        else => return error.InvalidInputPayload,
    }
    const input_path = request.input.asString() orelse return error.MissingInputPayload;
    const model_path = handle.normalized.artifacts.ocr_model_path orelse return error.MissingOCRModelArtifact;
    const context = swiftocr_native.Context{
        .operation = request.operation,
        .model_path = model_path,
        .input_path = input_path,
        .execution = request.execution,
    };
    const output = if (stateFromHandle(handle)) |state|
        if (state.model) |*model| try swiftocr_native.executeWithLoadedModel(allocator, context, model) else try swiftocr_native.execute(allocator, context)
    else
        try swiftocr_native.execute(allocator, context);
    return .{
        .origin = .runtime_backend,
        .note = .ocr_swiftocr_native,
        .output = .{ .json = output },
    };
}

fn stateFromHandle(handle: *const handle_mod.ModelHandle) ?*State {
    const raw = handle.provider_state orelse return null;
    return @ptrCast(@alignCast(raw));
}
