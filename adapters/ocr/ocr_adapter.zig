const std = @import("std");
const kinetix = @import("../../engine/kinetix.zig");

const adapter_mod = kinetix.adapter;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const legacy_command = @import("../legacy_command.zig");
const registry_mod = kinetix.registry;
const task = kinetix.core.task;

const swiftocr_operations = [_][]const u8{ "infer-ocr", "detect-text", "recognize-text" };

pub const OCRAdapter = struct {
    allocator: std.mem.Allocator,
    catalog: backend.ModelCatalog,
    plan: load_plan.ResolvedLoadPlan,
    adapter_id: []u8,
    descriptor: adapter_mod.Descriptor,

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8) !OCRAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        errdefer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
        });
        if (plan.ocr_model_path == null) return error.MissingOCRModelArtifact;

        const basename = std.fs.path.basename(catalog.model_dir);
        const adapter_id = try std.fmt.allocPrint(allocator, "ocr.swiftocr.{s}", .{basename});
        errdefer allocator.free(adapter_id);

        return .{
            .allocator = allocator,
            .catalog = catalog,
            .plan = plan,
            .adapter_id = adapter_id,
            .descriptor = .{
                .id = adapter_id,
                .modality = .ocr,
                .bound_model_family = "swiftocr",
                .supports_batching = false,
                .supports_streaming = false,
                .supported_operations = &swiftocr_operations,
            },
        };
    }

    pub fn deinit(self: *OCRAdapter) void {
        self.allocator.free(self.adapter_id);
        self.catalog.deinit();
        self.* = undefined;
    }

    pub fn asAdapter(self: *OCRAdapter) adapter_mod.Adapter {
        return .{
            .ctx = self,
            .descriptor = self.descriptor,
            .vtable = &vtable,
        };
    }

    pub fn registerInto(self: *OCRAdapter, registry: *registry_mod.Registry) !void {
        try registry.register(self.asAdapter());
    }

    pub fn buildLegacyCommand(self: *const OCRAdapter, allocator: std.mem.Allocator, options: legacy_command.BuildOptions) !legacy_command.LegacyCommand {
        const project_dir = try legacy_command.legacyProjectDirAlloc(allocator, "legacy/swiftocr");
        defer allocator.free(project_dir);

        return try legacy_command.init(allocator, project_dir, &.{
            "zig", "build", "run", "--", "infer",
            "--model", self.plan.ocr_model_path.?,
            "--image", options.input orelse "input.ppm",
        });
    }

    fn submit(ctx: *anyopaque, request: task.TaskRequest) !adapter_mod.Submission {
        const self: *OCRAdapter = @ptrCast(@alignCast(ctx));
        if (self.plan.ocr_model_path == null) return error.MissingOCRModelArtifact;
        switch (request.input) {
            .none, .image_path => {},
            else => return error.InvalidInputPayload,
        }

        return .{
            .adapter_id = self.descriptor.id,
            .accepted = true,
            .execution = request.spec.execution,
        };
    }
};

const vtable = adapter_mod.VTable{
    .submit = OCRAdapter.submit,
};
