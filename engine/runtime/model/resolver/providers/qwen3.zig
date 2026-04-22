const std = @import("std");
const catalog_mod = @import("../../../catalog/catalog.zig");
const common = @import("common.zig");
const normalized = @import("../normalized_model.zig");
const report_mod = @import("../support_report.zig");
const types = @import("../../../types.zig");

const operations = [_][]const u8{ "generate", "chat", "embed" };
const accepted_inputs = [_]types.InputKind{.text};

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    if (!catalog.has(.config) or !catalog.has(.tokenizer_json)) return null;
    if (catalog.resolveAutoScheme() == .auto and !catalog.has(.safetensors)) return null;

    const config_path = catalog.find(.config).?.absolute_path;
    const model_type = try common.readModelTypeAlloc(allocator, config_path) orelse return null;
    defer allocator.free(model_type);
    if (!std.mem.eql(u8, model_type, "qwen3")) return null;

    const selection = try catalog.resolveWeights(preferred_weights);
    const basename = std.fs.path.basename(catalog.modelDir());
    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.text.qwen3.{s}", .{basename}),
        .modality = .text,
        .family = try allocator.dupe(u8, "qwen3"),
        .source_format = .huggingface_directory,
        .normalized_format = .text_decoder,
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

    const support = try report_mod.RuntimeSupportReport.init(
        allocator,
        .supported,
        &.{},
        &.{.quantized_weights_selected},
    );
    errdefer {
        var owned = support;
        owned.deinit();
    }

    return normalized.NormalizedModel{
        .descriptor = descriptor,
        .artifacts = artifacts,
        .capabilities = .{
            .supports_sync = true,
            .supports_async = false,
            .supports_stream = true,
            .supports_batch = true,
            .supports_native_exec = true,
            .supported_operations = &operations,
            .accepted_inputs = &accepted_inputs,
        },
        .support = support,
        .provider_key = .qwen3_text,
    };
}
