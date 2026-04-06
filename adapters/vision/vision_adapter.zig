const std = @import("std");
const builtin = @import("builtin");
const kinetix = @import("../../engine/kinetix.zig");

const adapter_mod = kinetix.adapter;
const graph = kinetix.artifacts.graph;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const legacy_command = @import("../legacy_command.zig");
const registry_mod = kinetix.registry;
const task = kinetix.core.task;
const vision_legacy_bridge = if (builtin.is_test) struct {
    pub const Detection = struct {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        score: f64,
        class_id: usize,
    };

    pub const DetectOutput = struct {
        candidate_count: usize,
        detections: []Detection,

        pub fn deinit(self: *DetectOutput, allocator: std.mem.Allocator) void {
            allocator.free(self.detections);
            self.* = undefined;
        }
    };

    pub fn executeDetectJson(
        allocator: std.mem.Allocator,
        graph_path: []const u8,
        weights_path: []const u8,
        image_path: []const u8,
    ) !DetectOutput {
        _ = graph_path;
        _ = weights_path;
        _ = image_path;
        const detections = try allocator.alloc(Detection, 1);
        detections[0] = .{
            .x1 = 1.0,
            .y1 = 2.0,
            .x2 = 3.0,
            .y2 = 4.0,
            .score = 0.95,
            .class_id = 1,
        };
        return .{
            .candidate_count = 4,
            .detections = detections,
        };
    }
} else struct {
    pub const Detection = struct {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        score: f64,
        class_id: usize,
    };

    pub const DetectOutput = struct {
        candidate_count: usize,
        detections: []Detection,

        pub fn deinit(self: *DetectOutput, allocator: std.mem.Allocator) void {
            allocator.free(self.detections);
            self.* = undefined;
        }
    };

    const ParsedDetectOutput = struct {
        candidate_count: usize,
        detections: []Detection,
    };

    pub fn executeDetectJson(
        allocator: std.mem.Allocator,
        graph_path: []const u8,
        weights_path: []const u8,
        image_path: []const u8,
    ) !DetectOutput {
        const resolved_graph_path = if (std.fs.path.isAbsolute(graph_path))
            try allocator.dupe(u8, graph_path)
        else
            try std.fs.cwd().realpathAlloc(allocator, graph_path);
        defer allocator.free(resolved_graph_path);

        const resolved_weights_path = if (std.fs.path.isAbsolute(weights_path))
            try allocator.dupe(u8, weights_path)
        else
            try std.fs.cwd().realpathAlloc(allocator, weights_path);
        defer allocator.free(resolved_weights_path);

        const resolved_image_path = if (std.fs.path.isAbsolute(image_path))
            try allocator.dupe(u8, image_path)
        else
            try std.fs.cwd().realpathAlloc(allocator, image_path);
        defer allocator.free(resolved_image_path);

        const tmp_dir = std.process.getEnvVarOwned(allocator, "TEMP") catch try allocator.dupe(u8, ".");
        defer allocator.free(tmp_dir);

        const stamp = std.time.microTimestamp();
        const json_name = try std.fmt.allocPrint(allocator, "kinetix_vision_detect_{d}.json", .{stamp});
        defer allocator.free(json_name);
        const json_path = try std.fs.path.join(allocator, &.{ tmp_dir, json_name });
        defer allocator.free(json_path);

        const workdir = try legacy_command.legacyProjectDirAlloc(allocator, "legacy/axionyx");
        defer allocator.free(workdir);

        var command = try legacy_command.init(allocator, workdir, &.{
            "zig", "build", "run", "--",
            resolved_graph_path,
            resolved_weights_path,
            resolved_image_path,
            json_path,
        });
        defer command.deinit();

        var child = std.process.Child.init(command.argv, allocator);
        child.cwd = command.workdir;
        child.stdin_behavior = .Ignore;
        child.stdout_behavior = .Ignore;
        child.stderr_behavior = .Ignore;
        try child.spawn();
        const term = try child.wait();
        switch (term) {
            .Exited => |code| if (code != 0) return error.LegacyVisionDetectFailed,
            else => return error.LegacyVisionDetectFailed,
        }

        defer if (std.fs.path.isAbsolute(json_path))
            std.fs.deleteFileAbsolute(json_path) catch {}
        else
            std.fs.cwd().deleteFile(json_path) catch {};

        const bytes = if (std.fs.path.isAbsolute(json_path))
            blk: {
                const file = try std.fs.openFileAbsolute(json_path, .{});
                defer file.close();
                break :blk try file.readToEndAlloc(allocator, 4 * 1024 * 1024);
            }
        else
            try std.fs.cwd().readFileAlloc(allocator, json_path, 4 * 1024 * 1024);
        defer allocator.free(bytes);

        const parsed = try std.json.parseFromSlice(ParsedDetectOutput, allocator, bytes, .{});
        defer parsed.deinit();

        const detections = try allocator.alloc(Detection, parsed.value.detections.len);
        for (parsed.value.detections, detections) |det, *owned| owned.* = det;

        return .{
            .candidate_count = parsed.value.candidate_count,
            .detections = detections,
        };
    }
};

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
        const maybe_detection_output = try maybeRunLegacyDetect(self, allocator, request);
        var detection_output = maybe_detection_output;
        defer if (detection_output) |*output| output.deinit(allocator);

        const output = try buildOutputJson(self, allocator, request, detection_output);
        return .{
            .submission = try submit(ctx, request),
            .origin = if (detection_output != null) .legacy_process_bridge else .shared_adapter,
            .note = if (detection_output != null) .vision_legacy_detect_json else .vision_graph_ready,
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

fn buildOutputJson(
    self: *const VisionAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
    detection_output: ?vision_legacy_bridge.DetectOutput,
) ![]u8 {
    const Detection = vision_legacy_bridge.Detection;
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

fn maybeRunLegacyDetect(
    self: *const VisionAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
) !?vision_legacy_bridge.DetectOutput {
    if (!std.mem.eql(u8, request.spec.operation, "detect")) return null;
    if (request.spec.execution != .sync) return null;

    const image_path = switch (request.input) {
        .image_path => |value| value,
        else => return null,
    };

    return try vision_legacy_bridge.executeDetectJson(
        allocator,
        self.plan.graph_path.?,
        self.plan.binary_weights_path.?,
        image_path,
    );
}
