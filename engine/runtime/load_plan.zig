const backend = @import("../artifacts/backend/backend.zig");

pub const ResolvedLoadPlan = struct {
    model_dir: []const u8,
    weight_scheme: ?backend.WeightScheme,
    weights_path: ?[]const u8,
    graph_path: ?[]const u8,
    binary_weights_path: ?[]const u8,
    ocr_model_path: ?[]const u8,
    config_path: ?[]const u8,
    tokenizer_path: ?[]const u8,
};

pub const LoadRequest = struct {
    model_dir: []const u8,
    preferred_weights: backend.WeightScheme = .auto,
};

pub fn resolve(catalog: *const backend.ModelCatalog, request: LoadRequest) !ResolvedLoadPlan {
    var plan = ResolvedLoadPlan{
        .model_dir = request.model_dir,
        .weight_scheme = null,
        .weights_path = null,
        .graph_path = null,
        .binary_weights_path = null,
        .ocr_model_path = null,
        .config_path = null,
        .tokenizer_path = null,
    };

    if (catalog.find(.config)) |artifact| {
        plan.config_path = artifact.absolute_path;
    }

    if (catalog.find(.tokenizer_json)) |artifact| {
        plan.tokenizer_path = artifact.absolute_path;
    } else if (catalog.find(.tokenizer_model)) |artifact| {
        plan.tokenizer_path = artifact.absolute_path;
    } else if (catalog.find(.vocab_json)) |artifact| {
        plan.tokenizer_path = artifact.absolute_path;
    } else if (catalog.find(.vocab_txt)) |artifact| {
        plan.tokenizer_path = artifact.absolute_path;
    }

    if (catalog.find(.graph_json)) |artifact| {
        plan.graph_path = artifact.absolute_path;
    }

    if (catalog.find(.weights_bin)) |artifact| {
        plan.binary_weights_path = artifact.absolute_path;
    }

    if (catalog.find(.ocr_model)) |artifact| {
        plan.ocr_model_path = artifact.absolute_path;
    }

    if (catalog.resolveAutoScheme() != .auto or request.preferred_weights != .auto) {
        const selection = try catalog.resolveWeights(request.preferred_weights);
        plan.weight_scheme = selection.scheme;
        plan.weights_path = selection.path;
    }

    return plan;
}
