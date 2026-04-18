const std = @import("std");
const catalog_mod = @import("../../catalog/catalog.zig");
const normalized = @import("../normalized_model.zig");
const report_mod = @import("../capability_report.zig");
const types = @import("../../types.zig");

const operations = [_][]const u8{"infer-ocr"};
const accepted_inputs = [_]types.InputKind{.image_path};

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    _ = preferred_weights;
    if (!catalog.has(.ocr_model)) return null;

    const basename = std.fs.path.basename(catalog.modelDir());
    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.ocr.swiftocr.{s}", .{basename}),
        .modality = .ocr,
        .family = try allocator.dupe(u8, "swiftocr"),
        .source_format = .swiftocr_bundle,
        .normalized_format = .ocr_bundle,
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

    const compat = try report_mod.CompatibilityReport.init(
        allocator,
        .supported,
        &.{},
        &.{.ocr_single_file_bundle},
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
        .provider_key = .swiftocr_ocr,
    };
}
