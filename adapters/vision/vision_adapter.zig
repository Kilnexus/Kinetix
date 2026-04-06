const std = @import("std");
const kinetix = @import("../../engine/kinetix.zig");

const adapter_mod = kinetix.adapter;
const graph = kinetix.artifacts.graph;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const registry_mod = kinetix.registry;
const task = kinetix.core.task;
const axionyx_bridge = @import("axionyx_bridge.zig");

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

        return .{
            .allocator = allocator,
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
        const maybe_detection_output = try maybeRunSharedDetect(self, allocator, request);
        var detection_output = maybe_detection_output;
        defer if (detection_output) |*output| output.deinit(allocator);

        const output = try buildOutputJson(self, allocator, request, detection_output);
        return .{
            .submission = try submit(ctx, request),
            .origin = .shared_adapter,
            .note = if (detection_output != null) .vision_shared_detect else .vision_graph_ready,
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

fn buildOutputJson(
    self: *const VisionAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
    detection_output: ?axionyx_bridge.DetectOutput,
) ![]u8 {
    const Detection = axionyx_bridge.Detection;
    const VisionReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_name: []const u8,
        model_family: []const u8,
        input_path: ?[]const u8,
        execution_nodes: usize,
        tensor_count: usize,
        class_count: ?usize,
        candidate_count: ?usize,
        detections: []const Detection,
    };

    const receipt = VisionReceipt{
        .status = if (detection_output != null) "detect_completed" else "graph_ready",
        .operation = request.spec.operation,
        .model_name = self.graph_summary.model_name,
        .model_family = self.family.name(),
        .input_path = request.input.asString(),
        .execution_nodes = self.graph_summary.execution_nodes,
        .tensor_count = self.graph_summary.tensor_count,
        .class_count = self.graph_summary.class_count,
        .candidate_count = if (detection_output) |output| output.candidate_count else null,
        .detections = if (detection_output) |output| output.detections else &.{},
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn maybeRunSharedDetect(
    self: *const VisionAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
) !?axionyx_bridge.DetectOutput {
    if (!std.mem.eql(u8, request.spec.operation, "detect")) return null;
    if (request.spec.execution != .sync) return null;

    const image_path = switch (request.input) {
        .image_path => |value| value,
        else => return null,
    };

    return try axionyx_bridge.executeDetect(
        allocator,
        self.plan.graph_path.?,
        self.plan.binary_weights_path.?,
        image_path,
    );
}
