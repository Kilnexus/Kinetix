const std = @import("std");
const backend = @import("../artifacts/backend/backend.zig");

pub const ResolvedLoadPlan = struct {
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    weight_scheme: ?backend.WeightScheme,
    weights_path: ?[]const u8,
    graph_path: ?[]const u8,
    binary_weights_path: ?[]const u8,
    ocr_model_path: ?[]const u8,
    config_path: ?[]const u8,
    tokenizer_path: ?[]const u8,

    pub fn deinit(self: *ResolvedLoadPlan) void {
        self.allocator.free(@constCast(self.model_dir));
        freeOptionalPath(self.allocator, self.weights_path);
        freeOptionalPath(self.allocator, self.graph_path);
        freeOptionalPath(self.allocator, self.binary_weights_path);
        freeOptionalPath(self.allocator, self.ocr_model_path);
        freeOptionalPath(self.allocator, self.config_path);
        freeOptionalPath(self.allocator, self.tokenizer_path);
        self.* = undefined;
    }
};

pub const LoadRequest = struct {
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme = .auto,
};

pub fn resolve(catalog: *const backend.ModelCatalog, request: LoadRequest) !ResolvedLoadPlan {
    var plan = ResolvedLoadPlan{
        .allocator = catalog.allocator,
        .model_dir = try catalog.allocator.dupe(u8, request.model_dir),
        .weight_scheme = null,
        .weights_path = null,
        .graph_path = null,
        .binary_weights_path = null,
        .ocr_model_path = null,
        .config_path = null,
        .tokenizer_path = null,
    };
    errdefer plan.deinit();

    if (catalog.find(.config)) |artifact| {
        plan.config_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    }

    if (catalog.find(.tokenizer_json)) |artifact| {
        plan.tokenizer_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    } else if (catalog.find(.tokenizer_model)) |artifact| {
        plan.tokenizer_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    } else if (catalog.find(.vocab_json)) |artifact| {
        plan.tokenizer_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    } else if (catalog.find(.vocab_txt)) |artifact| {
        plan.tokenizer_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    }

    if (catalog.find(.graph_json)) |artifact| {
        plan.graph_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    }

    if (catalog.find(.weights_bin)) |artifact| {
        plan.binary_weights_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    }

    if (catalog.find(.ocr_model)) |artifact| {
        plan.ocr_model_path = try catalog.allocator.dupe(u8, artifact.absolute_path);
    }

    if (catalog.resolveAutoScheme() != .auto or request.preferred_weights != .auto) {
        const selection = try catalog.resolveWeights(request.preferred_weights);
        plan.weight_scheme = selection.scheme;
        plan.weights_path = try catalog.allocator.dupe(u8, selection.path);
    }

    return plan;
}

fn freeOptionalPath(allocator: std.mem.Allocator, value: ?[]const u8) void {
    if (value) |path| allocator.free(@constCast(path));
}
