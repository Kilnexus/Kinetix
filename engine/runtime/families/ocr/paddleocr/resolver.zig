const std = @import("std");
const catalog_mod = @import("../../../catalog/catalog.zig");
const normalized = @import("../../../model/resolver/normalized_model.zig");
const report_mod = @import("../../../model/resolver/support_report.zig");
const types = @import("../../../types.zig");

const io = std.Options.debug_io;

const operations = [_][]const u8{"infer-ocr"};
const operation_ids = [_]types.RuntimeOperation{.ocr};
const accepted_inputs = [_]types.InputKind{ .image_path, .document_path };

pub const Layout = struct {
    has_paddle_inference_model: bool = false,
    has_paddle_inference_params: bool = false,
    has_inference_yml: bool = false,
    has_model_yml: bool = false,
    has_rec_dict: bool = false,
    has_onnx_model: bool = false,
    det_model_count: usize = 0,
    rec_model_count: usize = 0,
    cls_model_count: usize = 0,

    pub fn isPaddleOCR(self: Layout) bool {
        if (self.has_paddle_inference_model and self.has_paddle_inference_params) return true;
        if (self.has_onnx_model and (self.has_inference_yml or self.has_model_yml or self.has_rec_dict)) return true;
        if (self.det_model_count != 0 or self.rec_model_count != 0 or self.cls_model_count != 0) return true;
        return false;
    }
};

pub fn tryNormalize(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
    preferred_weights: types.WeightScheme,
) !?normalized.NormalizedModel {
    _ = preferred_weights;

    const layout = inspectLayout(catalog.modelDir());
    if (!layout.isPaddleOCR()) return null;

    const basename = std.fs.path.basename(catalog.modelDir());
    const descriptor = normalized.RuntimeModelDescriptor{
        .allocator = allocator,
        .id = try std.fmt.allocPrint(allocator, "runtime.ocr.paddleocr.{s}", .{basename}),
        .modality = .ocr,
        .family = try allocator.dupe(u8, "paddleocr"),
        .variant = try allocator.dupe(u8, if (layout.has_onnx_model) "pp-ocr-onnx" else "pp-ocr-paddle-inference"),
        .source_format = if (layout.has_onnx_model) .onnx_bundle else .unknown,
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

    const support = try report_mod.RuntimeSupportReport.init(
        allocator,
        .degraded,
        &.{.document_input_partial},
        &.{.graph_schema_accepted},
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
        .provider_key = .paddleocr_ocr,
    };
}

pub fn inspectLayout(model_dir: []const u8) Layout {
    var layout = Layout{};
    inspectDir(model_dir, "", &layout, 0) catch {};
    return layout;
}

fn inspectDir(root: []const u8, relative: []const u8, layout: *Layout, depth: usize) !void {
    if (depth > 3) return;

    const dir_path = if (relative.len == 0)
        try std.heap.page_allocator.dupe(u8, root)
    else
        try std.fs.path.join(std.heap.page_allocator, &.{ root, relative });
    defer std.heap.page_allocator.free(dir_path);

    var dir = if (std.fs.path.isAbsolute(dir_path))
        std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return
    else
        std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        switch (entry.kind) {
            .file => inspectFile(relative, entry.name, layout),
            .directory => {
                const child = if (relative.len == 0)
                    try std.heap.page_allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(std.heap.page_allocator, &.{ relative, entry.name });
                defer std.heap.page_allocator.free(child);
                try inspectDir(root, child, layout, depth + 1);
            },
            else => {},
        }
    }
}

fn inspectFile(relative: []const u8, name: []const u8, layout: *Layout) void {
    if (std.mem.eql(u8, name, "inference.pdmodel")) layout.has_paddle_inference_model = true;
    if (std.mem.eql(u8, name, "inference.pdiparams")) layout.has_paddle_inference_params = true;
    if (std.mem.eql(u8, name, "inference.yml") or std.mem.eql(u8, name, "inference.yaml")) layout.has_inference_yml = true;
    if (std.mem.eql(u8, name, "model.yml") or std.mem.eql(u8, name, "model.yaml")) layout.has_model_yml = true;
    if (std.mem.indexOf(u8, name, "dict") != null and std.mem.endsWith(u8, name, ".txt")) layout.has_rec_dict = true;
    if (std.mem.endsWith(u8, name, ".onnx")) layout.has_onnx_model = true;

    const path = if (relative.len == 0) name else relative;
    if (std.ascii.indexOfIgnoreCase(path, "det") != null and isModelFile(name)) layout.det_model_count += 1;
    if (std.ascii.indexOfIgnoreCase(path, "rec") != null and isModelFile(name)) layout.rec_model_count += 1;
    if (std.ascii.indexOfIgnoreCase(path, "cls") != null and isModelFile(name)) layout.cls_model_count += 1;
}

fn isModelFile(name: []const u8) bool {
    return std.mem.eql(u8, name, "inference.pdmodel") or std.mem.endsWith(u8, name, ".onnx");
}
