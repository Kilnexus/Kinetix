const std = @import("std");
const catalog_mod = @import("../../catalog/catalog.zig");
const normalized = @import("../normalized_model.zig");
const report_mod = @import("../capability_report.zig");
const provider_common = @import("common.zig");
const types = @import("../../types.zig");

const operations = [_][]const u8{
    "infer-ocr",
    "render-markdown",
    "render-html",
    "render-json",
};
const accepted_inputs = [_]types.InputKind{ .image_path, .document_path };

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    if (!catalog.has(.config)) return null;

    const basename = std.fs.path.basename(catalog.modelDir());
    if (std.ascii.indexOfIgnoreCase(basename, "chandra") == null) return null;

    const config_path = catalog.find(.config).?.absolute_path;
    const model_type = try provider_common.readModelTypeAlloc(allocator, config_path);
    defer if (model_type) |owned| allocator.free(owned);
    if (model_type == null or !std.mem.eql(u8, model_type.?, "qwen3_5")) return null;

    if (!catalog.has(.tokenizer_json)) return null;
    if (catalog.resolveAutoScheme() == .auto and !catalog.has(.safetensors)) return null;
    const selection = try catalog.resolveWeights(preferred_weights);

    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.ocr.chandra.{s}", .{basename}),
        .modality = .ocr,
        .family = try allocator.dupe(u8, "chandra"),
        .variant = try allocator.dupe(u8, basename),
        .source_format = .huggingface_directory,
        .normalized_format = .document_vlm,
    };
    errdefer {
        var owned = descriptor;
        owned.deinit();
    }

    const artifacts = try normalized.RuntimeArtifactSet.initFromCatalog(allocator, catalog, .{
        .selected_weights_path = selection.path,
    });
    errdefer {
        var owned = artifacts;
        owned.deinit();
    }

    const compat = try report_mod.CompatibilityReport.init(
        allocator,
        .degraded,
        &.{.external_runtime_required},
        &.{},
    );
    errdefer {
        var owned = compat;
        owned.deinit();
    }

    return normalized.NormalizedModel{
        .descriptor = descriptor,
        .artifacts = artifacts,
        .capabilities = .{
            .supports_sync = true,
            .supports_async = false,
            .supports_stream = false,
            .supports_batch = false,
            .supports_native_exec = true,
            .supported_operations = &operations,
            .accepted_inputs = &accepted_inputs,
        },
        .compat = compat,
        .provider_key = .chandra_ocr,
    };
}
