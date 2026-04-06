const std = @import("std");
const kinetix = @import("../../engine/kinetix.zig");

const adapter_mod = kinetix.adapter;
const graph = kinetix.artifacts.graph;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const legacy_command = @import("../legacy_command.zig");
const registry_mod = kinetix.registry;
const task = kinetix.core.task;

const yolo_operations = [_][]const u8{ "detect", "profile", "benchmark" };
const generic_operations = [_][]const u8{ "infer-image" };

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
    catalog: backend.ModelCatalog,
    plan: load_plan.ResolvedLoadPlan,
    family: ModelFamily,
    adapter_id: []u8,
    graph_summary: graph.Summary,
    descriptor: adapter_mod.Descriptor,

    pub fn init(allocator: std.mem.Allocator, model_dir: []const u8) !VisionAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        errdefer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
        });
        if (plan.graph_path == null) return error.MissingGraphArtifact;
        if (plan.binary_weights_path == null) return error.MissingBinaryWeightsArtifact;

        const summary = try graph.loadSummary(allocator, plan.graph_path.?);
        errdefer allocator.free(summary.model_name);

        const family = try detectModelFamily(allocator, plan.graph_path.?);
        const basename = std.fs.path.basename(catalog.model_dir);
        const adapter_id = try std.fmt.allocPrint(allocator, "vision.{s}.{s}", .{ family.name(), basename });
        errdefer allocator.free(adapter_id);

        return .{
            .allocator = allocator,
            .catalog = catalog,
            .plan = plan,
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
        self.catalog.deinit();
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

    pub fn buildLegacyCommand(self: *const VisionAdapter, allocator: std.mem.Allocator, options: legacy_command.BuildOptions) !legacy_command.LegacyCommand {
        const project_dir = try legacy_command.legacyProjectDirAlloc(allocator, "legacy/axionyx");
        defer allocator.free(project_dir);

        const input = options.input orelse defaultImagePath(self.catalog.model_dir);
        if (std.mem.eql(u8, options.operation, "detect")) {
            return try legacy_command.init(allocator, project_dir, &.{
                "zig", "build", "run", "--",
                self.plan.graph_path.?,
                self.plan.binary_weights_path.?,
                input,
            });
        }
        if (std.mem.eql(u8, options.operation, "profile")) {
            return try legacy_command.init(allocator, project_dir, &.{
                "zig", "build", "run", "--",
                self.plan.graph_path.?,
                self.plan.binary_weights_path.?,
                "profile",
                input,
            });
        }
        if (std.mem.eql(u8, options.operation, "benchmark")) {
            return try legacy_command.init(allocator, project_dir, &.{
                "zig", "build", "run", "--",
                self.plan.graph_path.?,
                self.plan.binary_weights_path.?,
                "bench",
                input,
            });
        }
        return error.UnsupportedLegacyOperation;
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
        const output = try buildOutputJson(self, allocator, request);
        return .{
            .submission = try submit(ctx, request),
            .origin = .shared_adapter,
            .note = .vision_graph_ready,
            .output = .{ .json = output },
        };
    }
};

const vtable = adapter_mod.VTable{
    .submit = VisionAdapter.submit,
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

fn defaultImagePath(model_dir: []const u8) []const u8 {
    _ = model_dir;
    return "data/archive/images/000_0001.png";
}

fn buildOutputJson(self: *const VisionAdapter, allocator: std.mem.Allocator, request: task.TaskRequest) ![]u8 {
    const VisionReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_name: []const u8,
        model_family: []const u8,
        input_path: ?[]const u8,
        execution_nodes: usize,
        tensor_count: usize,
        class_count: ?usize,
        detections: []const struct {},
    };

    const receipt = VisionReceipt{
        .status = "graph_ready",
        .operation = request.spec.operation,
        .model_name = self.graph_summary.model_name,
        .model_family = self.family.name(),
        .input_path = request.input.asString(),
        .execution_nodes = self.graph_summary.execution_nodes,
        .tensor_count = self.graph_summary.tensor_count,
        .class_count = self.graph_summary.class_count,
        .detections = &.{},
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}
