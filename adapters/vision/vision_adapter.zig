const std = @import("std");
const kinetix = @import("engine_root");

const adapter_mod = kinetix.adapter;
const graph = kinetix.artifacts.graph;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const registry_mod = kinetix.registry;
const runtime_model = kinetix.runtime.model;
const runtime_providers = kinetix.runtime.providers;
const runtime_session = kinetix.runtime.session;
const task = kinetix.core.task;

const yolo_operations = [_][]const u8{ "detect", "profile", "benchmark" };
const generic_operations = [_][]const u8{"infer-image"};

pub const ModelFamily = enum {
    yolo,
    unknown,

    pub fn name(self: ModelFamily) []const u8 {
        return switch (self) {
            .yolo => "yolo",
            .unknown => "unknown",
        };
    }
};

pub const VisionAdapter = struct {
    allocator: std.mem.Allocator,
    plan: load_plan.ResolvedLoadPlan,
    runtime_handle: runtime_model.ModelHandle,
    family: ModelFamily,
    adapter_id: []u8,
    graph_summary: graph.Summary,
    descriptor: adapter_mod.Descriptor,

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8) !VisionAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        defer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
        });
        errdefer {
            var owned_plan = plan;
            owned_plan.deinit();
        }
        if (plan.graph_path == null) return error.MissingGraphArtifact;
        if (plan.binary_weights_path == null) return error.MissingBinaryWeightsArtifact;

        const summary = try graph.loadSummary(allocator, plan.graph_path.?);
        errdefer allocator.free(summary.model_name);

        const family = try detectModelFamily(allocator, plan.graph_path.?);
        const basename = std.fs.path.basename(plan.model_dir);
        const adapter_id = try std.fmt.allocPrint(allocator, "vision.{s}.{s}", .{ family.name(), basename });
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
            .family = family,
            .adapter_id = adapter_id,
            .graph_summary = summary,
            .descriptor = .{
                .id = adapter_id,
                .modality = .vision,
                .bound_model_family = family.name(),
                .supports_batching = true,
                .supports_streaming = false,
                .supported_operations = operationsForFamily(family),
            },
        };
    }

    pub fn deinit(self: *VisionAdapter) void {
        self.allocator.free(@constCast(self.graph_summary.model_name));
        self.allocator.free(self.adapter_id);
        self.runtime_handle.deinit();
        self.plan.deinit();
        self.* = undefined;
    }

    pub fn asAdapter(self: *VisionAdapter) adapter_mod.Adapter {
        return .{
            .ctx = self,
            .descriptor = self.descriptor,
            .vtable = &vtable,
        };
    }

    pub fn registerInto(self: *VisionAdapter, registry: *registry_mod.Registry) !void {
        try registry.register(self.asAdapter());
    }

    fn submit(ctx: *anyopaque, request: task.TaskRequest) !adapter_mod.Submission {
        const self: *VisionAdapter = @ptrCast(@alignCast(ctx));
        if (self.plan.graph_path == null) return error.MissingGraphArtifact;
        if (self.plan.binary_weights_path == null) return error.MissingBinaryWeightsArtifact;
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
        const self: *VisionAdapter = @ptrCast(@alignCast(ctx));
        return try runtime_providers.adapter_bridge.executeSingle(allocator, &self.runtime_handle, self.descriptor.id, request);
    }

    fn executeBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.ExecutionResult {
        const self: *VisionAdapter = @ptrCast(@alignCast(ctx));
        return try runtime_providers.adapter_bridge.executeBatch(allocator, &self.runtime_handle, self.descriptor.id, requests);
    }
};

const vtable = adapter_mod.VTable{
    .submit = VisionAdapter.submit,
    .execute_batch = VisionAdapter.executeBatch,
    .execute = VisionAdapter.execute,
};

fn operationsForFamily(family: ModelFamily) []const []const u8 {
    return switch (family) {
        .yolo => &yolo_operations,
        .unknown => &generic_operations,
    };
}

fn detectModelFamily(allocator: std.mem.Allocator, graph_path: []const u8) !ModelFamily {
    var parsed = try graph.load(allocator, graph_path);
    defer parsed.deinit();

    for (parsed.execution_nodes) |node| {
        if (std.mem.eql(u8, node.kind, "Detect")) return .yolo;
    }
    return .unknown;
}
