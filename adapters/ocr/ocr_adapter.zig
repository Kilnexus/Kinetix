const std = @import("std");
const kinetix = @import("engine_root");

const adapter_mod = kinetix.adapter;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const registry_mod = kinetix.registry;
const runtime_model = kinetix.runtime.model;
const runtime_providers = kinetix.runtime.providers;
const runtime_session = kinetix.runtime.session;
const task = kinetix.core.task;

const swiftocr_operations = [_][]const u8{ "infer-ocr", "detect-text", "recognize-text" };

pub const OCRAdapter = struct {
    allocator: std.mem.Allocator,
    plan: load_plan.ResolvedLoadPlan,
    runtime_handle: runtime_model.ModelHandle,
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
        var session = runtime_session.RuntimeSession.init(allocator);
        defer session.deinit();
        const runtime_handle = try session.openModel(.{ .model_dir = model_dir });
        errdefer {
            var owned = runtime_handle;
            owned.deinit();
        }

        return .{
            .allocator = allocator,
            .plan = plan,
            .runtime_handle = runtime_handle,
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
        self.runtime_handle.deinit();
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
        return try runtime_providers.adapter_bridge.executeSingle(allocator, &self.runtime_handle, self.descriptor.id, request);
    }

    fn executeBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.ExecutionResult {
        const self: *OCRAdapter = @ptrCast(@alignCast(ctx));
        return try runtime_providers.adapter_bridge.executeBatch(allocator, &self.runtime_handle, self.descriptor.id, requests);
    }
};

const vtable = adapter_mod.VTable{
    .submit = OCRAdapter.submit,
    .execute = OCRAdapter.execute,
    .execute_batch = OCRAdapter.executeBatch,
};
