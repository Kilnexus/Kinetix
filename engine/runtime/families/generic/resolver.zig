const std = @import("std");
const catalog_mod = @import("../../catalog/catalog.zig");
const normalized = @import("../../model/resolver/normalized_model.zig");
const report_mod = @import("../../model/resolver/support_report.zig");
const types = @import("../../types.zig");

const operations = [_][]const u8{"infer"};
const accepted_inputs = [_]types.InputKind{ .text, .image_path, .audio_path, .video_path };

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    _ = preferred_weights;
    if (catalog.artifactCount() == 0) return null;

    const basename = std.fs.path.basename(catalog.modelDir());
    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.generic.{s}", .{basename}),
        .modality = inferModality(catalog),
        .family = try allocator.dupe(u8, "generic"),
        .source_format = .unknown,
        .normalized_format = .generic,
    };
    errdefer {
        var owned = descriptor;
        owned.deinit();
    }

    const artifacts = try normalized.RuntimeArtifactSet.initFromCatalog(allocator, catalog, .{});
    errdefer {
        var owned = artifacts;
        owned.deinit();
    }

    const support = try report_mod.RuntimeSupportReport.init(
        allocator,
        .degraded,
        &.{.graph_runtime_backend_required},
        &.{},
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
            .supports_stream = false,
            .supports_batch = false,
            .supports_native_exec = false,
            .supported_operations = &operations,
            .accepted_inputs = &accepted_inputs,
        },
        .support = support,
        .provider_key = .generic,
    };
}

fn inferModality(catalog: *const catalog_mod.ArtifactCatalog) types.Modality {
    if (catalog.has(.graph_json)) return .vision;
    if (catalog.has(.ocr_model)) return .ocr;
    if (catalog.has(.config)) return .text;
    return .multimodal;
}
