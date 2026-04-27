const std = @import("std");
const bundle = @import("bundle/index.zig");
const catalog_mod = @import("../../../catalog/catalog.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const report_mod = @import("../../../model/resolver/support_report.zig");
const types = @import("../../../types.zig");

const operations = [_][]const u8{"synthesize"};
const operation_ids = [_]types.RuntimeOperation{.synthesize};
const accepted_inputs = [_]types.InputKind{.text};

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    _ = preferred_weights;

    var resolved = try bundle.paths.findPaths(allocator, catalog.modelDir()) orelse return null;
    defer resolved.deinit();

    const basename = std.fs.path.basename(catalog.modelDir());
    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.tts.moss_tts_nano.{s}", .{basename}),
        .modality = .tts,
        .family = try allocator.dupe(u8, "moss_tts_nano"),
        .source_format = .onnx_bundle,
        .normalized_format = .tts_onnx_bundle,
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
        &.{.tts_runtime_pending},
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
            .supported_operation_ids = &operation_ids,
            .accepted_inputs = &accepted_inputs,
        },
        .support = support,
        .provider_key = .moss_tts_nano_tts,
    };
}
