const std = @import("std");
const builtin = @import("builtin");
const kinetix = @import("../../engine/kinetix.zig");

const adapter_mod = kinetix.adapter;
const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const legacy_command = @import("../legacy_command.zig");
const registry_mod = kinetix.registry;
const task = kinetix.core.task;
const ocr_legacy_bridge = if (builtin.is_test) struct {
    pub const InferOutput = struct {
        loaded_tensors: usize,
        image_width: usize,
        image_height: usize,
    };

    pub fn executeInfer(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        image_path: []const u8,
    ) !InferOutput {
        _ = allocator;
        _ = model_path;
        _ = image_path;
        return .{
            .loaded_tensors = 2,
            .image_width = 1,
            .image_height = 1,
        };
    }
} else struct {
    pub const InferOutput = struct {
        loaded_tensors: usize,
        image_width: usize,
        image_height: usize,
    };

    pub fn executeInfer(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        image_path: []const u8,
    ) !InferOutput {
        const resolved_model_path = if (std.fs.path.isAbsolute(model_path))
            try allocator.dupe(u8, model_path)
        else
            try std.fs.cwd().realpathAlloc(allocator, model_path);
        defer allocator.free(resolved_model_path);

        const resolved_image_path = if (std.fs.path.isAbsolute(image_path))
            try allocator.dupe(u8, image_path)
        else
            try std.fs.cwd().realpathAlloc(allocator, image_path);
        defer allocator.free(resolved_image_path);

        const workdir = try legacy_command.legacyProjectDirAlloc(allocator, "legacy/swiftocr");
        defer allocator.free(workdir);

        const result = try std.process.Child.run(.{
            .allocator = allocator,
            .argv = &.{
                "zig",
                "build",
                "run",
                "--",
                "infer",
                "--model",
                resolved_model_path,
                "--image",
                resolved_image_path,
            },
            .cwd = workdir,
            .max_output_bytes = 64 * 1024,
        });
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        switch (result.term) {
            .Exited => |code| if (code != 0) return error.LegacyOCRInferFailed,
            else => return error.LegacyOCRInferFailed,
        }

        return try parseInferOutput(result.stdout);
    }

    fn parseInferOutput(stdout: []const u8) !InferOutput {
        const loaded_prefix = "Loaded tensors: ";
        const image_prefix = "Image: ";

        const loaded_start = std.mem.indexOf(u8, stdout, loaded_prefix) orelse return error.InvalidLegacyOCROutput;
        const image_start = std.mem.indexOf(u8, stdout, image_prefix) orelse return error.InvalidLegacyOCROutput;

        const loaded_line = sliceLine(stdout[loaded_start + loaded_prefix.len ..]);
        const image_line = sliceLine(stdout[image_start + image_prefix.len ..]);
        const x_index = std.mem.indexOfScalar(u8, image_line, 'x') orelse return error.InvalidLegacyOCROutput;

        return .{
            .loaded_tensors = try std.fmt.parseInt(usize, std.mem.trim(u8, loaded_line, " \r\n\t"), 10),
            .image_width = try std.fmt.parseInt(usize, std.mem.trim(u8, image_line[0..x_index], " \r\n\t"), 10),
            .image_height = try std.fmt.parseInt(usize, std.mem.trim(u8, image_line[x_index + 1 ..], " \r\n\t"), 10),
        };
    }

    fn sliceLine(bytes: []const u8) []const u8 {
        const newline_index = std.mem.indexOfScalar(u8, bytes, '\n') orelse bytes.len;
        return bytes[0..newline_index];
    }
};

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

        const model_path = if (std.fs.path.isAbsolute(self.plan.ocr_model_path.?))
            self.plan.ocr_model_path.?
        else
            try std.fs.cwd().realpathAlloc(allocator, self.plan.ocr_model_path.?);
        defer if (!std.fs.path.isAbsolute(self.plan.ocr_model_path.?)) allocator.free(model_path);

        const input_path = if (options.input) |value|
            if (std.fs.path.isAbsolute(value))
                value
            else
                try std.fs.cwd().realpathAlloc(allocator, value)
        else
            "input.ppm";
        defer if (options.input != null and !std.fs.path.isAbsolute(options.input.?)) allocator.free(input_path);

        return try legacy_command.init(allocator, project_dir, &.{
            "zig", "build", "run", "--", "infer",
            "--model", model_path,
            "--image", input_path,
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

    fn execute(ctx: *anyopaque, allocator: std.mem.Allocator, request: task.TaskRequest) !adapter_mod.ExecutionResult {
        const self: *OCRAdapter = @ptrCast(@alignCast(ctx));
        const infer_output = try maybeRunLegacyInfer(self, allocator, request);
        const output = try buildOutputJson(self, allocator, request, infer_output);
        return .{
            .submission = try submit(ctx, request),
            .origin = if (infer_output != null) .legacy_process_bridge else .shared_adapter,
            .note = if (infer_output != null) .ocr_legacy_infer_summary else .ocr_model_ready,
            .output = .{ .json = output },
        };
    }
};

const vtable = adapter_mod.VTable{
    .submit = OCRAdapter.submit,
    .execute = OCRAdapter.execute,
};

fn buildOutputJson(
    self: *const OCRAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
    infer_output: ?ocr_legacy_bridge.InferOutput,
) ![]u8 {
    const OCRReceipt = struct {
        status: []const u8,
        operation: []const u8,
        model_family: []const u8,
        model_path: []const u8,
        input_path: ?[]const u8,
        loaded_tensors: ?usize,
        image_width: ?usize,
        image_height: ?usize,
    };

    const receipt = OCRReceipt{
        .status = if (infer_output != null) "ocr_infer_completed" else "ocr_model_ready",
        .operation = request.spec.operation,
        .model_family = self.descriptor.bound_model_family.?,
        .model_path = self.plan.ocr_model_path.?,
        .input_path = request.input.asString(),
        .loaded_tensors = if (infer_output) |output| output.loaded_tensors else null,
        .image_width = if (infer_output) |output| output.image_width else null,
        .image_height = if (infer_output) |output| output.image_height else null,
    };

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(receipt, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

fn maybeRunLegacyInfer(
    self: *const OCRAdapter,
    allocator: std.mem.Allocator,
    request: task.TaskRequest,
) !?ocr_legacy_bridge.InferOutput {
    if (!std.mem.eql(u8, request.spec.operation, "infer-ocr")) return null;
    if (request.spec.execution != .sync) return null;

    const image_path = switch (request.input) {
        .image_path => |value| value,
        else => return null,
    };

    return try ocr_legacy_bridge.executeInfer(
        allocator,
        self.plan.ocr_model_path.?,
        image_path,
    );
}
