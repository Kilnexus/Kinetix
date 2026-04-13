const std = @import("std");
const kinetix = @import("engine_root");

const adapter_mod = kinetix.adapter;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const ocr_shared = kinetix.runtime.providers.ocr_shared;
const registry_mod = kinetix.registry;
const task = kinetix.core.task;

const swiftocr_operations = [_][]const u8{ "infer-ocr", "detect-text", "recognize-text" };

pub const OCRAdapter = struct {
    allocator: std.mem.Allocator,
    plan: load_plan.ResolvedLoadPlan,
    adapter_id: []u8,
    descriptor: adapter_mod.Descriptor,

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8) !OCRAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        defer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
        });
        errdefer {
            var owned_plan = plan;
            owned_plan.deinit();
        }
        if (plan.ocr_model_path == null) return error.MissingOCRModelArtifact;

        const basename = std.fs.path.basename(plan.model_dir);
        const adapter_id = try std.fmt.allocPrint(allocator, "ocr.swiftocr.{s}", .{basename});
        errdefer allocator.free(adapter_id);

        return .{
            .allocator = allocator,
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
        self.plan.deinit();
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

    fn execute(ctx: *anyopaque, allocator: std.mem.Allocator, request: task.TaskRequest) !adapter_mod.ExecutionResult {
        const self: *OCRAdapter = @ptrCast(@alignCast(ctx));
        const infer_output = try maybeRunSharedInfer(self, allocator, request);
        const output = try ocr_shared.buildOutputJson(allocator, .{
            .operation = request.spec.operation,
            .model_family = self.descriptor.bound_model_family.?,
            .model_path = self.plan.ocr_model_path.?,
            .input_path = request.input.asString(),
        }, infer_output);
        return .{
            .submission = try submit(ctx, request),
            .origin = .shared_adapter,
            .note = if (infer_output != null) .ocr_shared_infer else .ocr_model_ready,
            .output = .{ .json = output },
        };
    }
};

const vtable = adapter_mod.VTable{
    .submit = OCRAdapter.submit,
    .execute = OCRAdapter.execute,
};

fn maybeRunSharedInfer(
    self: *const OCRAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
) !?ocr_shared.InferResult {
    return try ocr_shared.maybeRunInfer(
        allocator,
        self.plan.ocr_model_path.?,
        request.spec.operation,
        request.spec.execution,
        request.input.asString(),
    );
}
