const std = @import("std");
const builtin = @import("builtin");
const kinetix = @import("../../engine/kinetix.zig");

const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const adapter_mod = kinetix.adapter;
const legacy_command = @import("../legacy_command.zig");
const registry_mod = kinetix.registry;
const task = kinetix.core.task;
const zinfer_batch_bridge = if (builtin.is_test) struct {
    pub fn executeQwenSingle(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
        request: task.TaskRequest,
    ) ![]u8 {
        _ = model_dir;
        _ = preferred_weights;
        _ = request;
        return try allocator.dupe(u8, "stub-native-single");
    }

    pub const NativeBatchOutput = struct {
        texts: [][]u8,
        total_decoded_tokens: usize,
        finished_requests: usize,

        pub fn deinit(self: *NativeBatchOutput, allocator: std.mem.Allocator) void {
            _ = allocator;
            _ = self;
        }
    };

    pub fn executeQwenBatch(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
        requests: []const task.TaskRequest,
    ) !NativeBatchOutput {
        _ = allocator;
        _ = model_dir;
        _ = preferred_weights;
        _ = requests;
        return error.NativeBatchBridgeUnavailableInTests;
    }
} else @import("zinfer_batch_bridge.zig");

pub const ModelFamily = enum {
    qwen3,
    bert,
    unknown,

    pub fn name(self: ModelFamily) []const u8 {
        return switch (self) {
            .qwen3 => "qwen3",
            .bert => "bert",
            .unknown => "unknown",
        };
    }
};

const qwen3_operations = [_][]const u8{ "generate", "chat", "embed" };
const bert_operations = [_][]const u8{ "fill-mask", "embed" };
const unknown_operations = [_][]const u8{ "infer-text" };

pub const TextAdapter = struct {
    allocator: std.mem.Allocator,
    catalog: backend.ModelCatalog,
    plan: load_plan.ResolvedLoadPlan,
    family: ModelFamily,
    adapter_id: []u8,
    descriptor: adapter_mod.Descriptor,

    pub fn init(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
    ) !TextAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        errdefer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
            .preferred_weights = preferred_weights,
        });

        if (plan.config_path == null) return error.MissingConfigArtifact;
        if (plan.tokenizer_path == null) return error.MissingTokenizerArtifact;
        if (plan.weights_path == null) return error.MissingWeightArtifacts;

        const family = try detectModelFamily(allocator, plan.config_path.?);
        const basename = std.fs.path.basename(catalog.model_dir);
        const family_name = family.name();
        const adapter_id = try std.fmt.allocPrint(allocator, "text.{s}.{s}", .{ family_name, basename });
        errdefer allocator.free(adapter_id);

        return .{
            .allocator = allocator,
            .catalog = catalog,
            .plan = plan,
            .family = family,
            .adapter_id = adapter_id,
            .descriptor = .{
                .id = adapter_id,
                .modality = .text,
                .bound_model_family = family_name,
                .supports_batching = true,
                .supports_streaming = family == .qwen3,
                .supported_operations = operationsForFamily(family),
            },
        };
    }

    pub fn deinit(self: *TextAdapter) void {
        self.allocator.free(self.adapter_id);
        self.catalog.deinit();
        self.* = undefined;
    }

    pub fn asAdapter(self: *TextAdapter) adapter_mod.Adapter {
        return .{
            .ctx = self,
            .descriptor = self.descriptor,
            .vtable = &vtable,
        };
    }

    pub fn registerInto(self: *TextAdapter, registry: *registry_mod.Registry) !void {
        try registry.register(self.asAdapter());
    }

    pub fn buildLegacyCommand(self: *const TextAdapter, allocator: std.mem.Allocator, options: legacy_command.BuildOptions) !legacy_command.LegacyCommand {
        const project_dir = try legacy_command.legacyProjectDirAlloc(allocator, "legacy/zinfer");
        defer allocator.free(project_dir);

        const input = options.input orelse defaultTextInput(self.family, options.operation);
        return switch (self.family) {
            .qwen3 => blk: {
                if (std.mem.eql(u8, options.operation, "generate") or std.mem.eql(u8, options.operation, "chat")) {
                    const max_tokens = options.max_tokens orelse 64;
                    const max_tokens_text = try std.fmt.allocPrint(allocator, "{d}", .{max_tokens});
                    defer allocator.free(max_tokens_text);
                    break :blk try legacy_command.init(allocator, project_dir, &.{
                        "zig", "build", "run", "--", "generate",
                        self.catalog.model_dir,
                        input,
                        max_tokens_text,
                    });
                }
                if (std.mem.eql(u8, options.operation, "embed")) {
                    break :blk try legacy_command.init(allocator, project_dir, &.{
                        "zig", "build", "run", "--", "embed-text",
                        self.catalog.model_dir,
                        input,
                    });
                }
                return error.UnsupportedLegacyOperation;
            },
            .bert => blk: {
                if (std.mem.eql(u8, options.operation, "fill-mask")) {
                    break :blk try legacy_command.init(allocator, project_dir, &.{
                        "zig", "build", "run", "--", "fill-mask",
                        self.catalog.model_dir,
                        input,
                    });
                }
                if (std.mem.eql(u8, options.operation, "embed")) {
                    break :blk try legacy_command.init(allocator, project_dir, &.{
                        "zig", "build", "run", "--", "embed-text",
                        self.catalog.model_dir,
                        input,
                    });
                }
                return error.UnsupportedLegacyOperation;
            },
            .unknown => return error.UnsupportedLegacyOperation,
        };
    }

    fn submit(ctx: *anyopaque, request: task.TaskRequest) !adapter_mod.Submission {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        if (self.plan.weights_path == null) return error.MissingWeightArtifacts;
        if (self.plan.config_path == null) return error.MissingConfigArtifact;
        if (self.plan.tokenizer_path == null) return error.MissingTokenizerArtifact;
        switch (request.input) {
            .none, .text => {},
            else => return error.InvalidInputPayload,
        }

        return .{
            .adapter_id = self.descriptor.id,
            .accepted = true,
            .execution = request.spec.execution,
        };
    }

    fn submitBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.Submission {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        if (self.plan.weights_path == null) return error.MissingWeightArtifacts;
        if (self.plan.config_path == null) return error.MissingConfigArtifact;
        if (self.plan.tokenizer_path == null) return error.MissingTokenizerArtifact;
        return try buildSubmissions(self, allocator, requests);
    }

    fn execute(ctx: *anyopaque, allocator: std.mem.Allocator, request: task.TaskRequest) !adapter_mod.ExecutionResult {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        const use_native = canUseNativeQwenSingle(self, request);
        const output = if (use_native)
            adapter_mod.OutputPayload{ .text = try zinfer_batch_bridge.executeQwenSingle(
                allocator,
                self.catalog.model_dir,
                self.plan.weight_scheme orelse .auto,
                request,
            ) }
        else
            .none;
        return .{
            .submission = try submit(ctx, request),
            .origin = if (use_native) .native_single_bridge else .shared_adapter,
            .note = if (use_native) .text_native_qwen_single else .text_request_ready,
            .output = output,
        };
    }

    fn executeBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.ExecutionResult {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        const use_native = canUseNativeQwenBatch(self, requests);
        var native_output: ?zinfer_batch_bridge.NativeBatchOutput = null;
        defer if (native_output) |*output| output.deinit(allocator);

        if (use_native) {
            native_output = try zinfer_batch_bridge.executeQwenBatch(
                allocator,
                self.catalog.model_dir,
                self.plan.weight_scheme orelse .auto,
                requests,
            );
        }

        const results = try allocator.alloc(adapter_mod.ExecutionResult, requests.len);
        errdefer allocator.free(results);

        for (requests, results, 0..) |request, *result, index| {
            result.* = .{
                .submission = buildSubmission(self, request),
                .origin = if (use_native) .native_batch_bridge else .shared_adapter,
                .note = if (use_native) .text_native_qwen_batch else .text_request_ready,
                .output = if (use_native)
                    .{ .text = try allocator.dupe(u8, native_output.?.texts[index]) }
                else
                    .none,
            };
        }

        return results;
    }
};

const vtable = adapter_mod.VTable{
    .submit = TextAdapter.submit,
    .submit_batch = TextAdapter.submitBatch,
    .execute = TextAdapter.execute,
    .execute_batch = TextAdapter.executeBatch,
};

fn operationsForFamily(family: ModelFamily) []const []const u8 {
    return switch (family) {
        .qwen3 => &qwen3_operations,
        .bert => &bert_operations,
        .unknown => &unknown_operations,
    };
}

fn detectModelFamily(allocator: std.mem.Allocator, config_path: []const u8) !ModelFamily {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, config_path, 1024 * 1024);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();

    const model_type_value = parsed.value.object.get("model_type") orelse return .unknown;
    if (model_type_value != .string) return error.InvalidModelType;

    if (std.mem.eql(u8, model_type_value.string, "qwen3")) return .qwen3;
    if (std.mem.eql(u8, model_type_value.string, "bert")) return .bert;
    return .unknown;
}

fn defaultTextInput(family: ModelFamily, operation: []const u8) []const u8 {
    if (family == .bert and std.mem.eql(u8, operation, "fill-mask")) return "Hello [MASK]";
    return "Hello from Kinetix";
}

fn buildSubmissions(self: *const TextAdapter, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.Submission {
    const submissions = try allocator.alloc(adapter_mod.Submission, requests.len);
    errdefer allocator.free(submissions);

    for (requests, submissions) |request, *submission| {
        switch (request.input) {
            .none, .text => {},
            else => return error.InvalidInputPayload,
        }

        submission.* = buildSubmission(self, request);
    }

    return submissions;
}

fn buildSubmission(self: *const TextAdapter, request: task.TaskRequest) adapter_mod.Submission {
    return .{
        .adapter_id = self.descriptor.id,
        .accepted = true,
        .execution = request.spec.execution,
    };
}

fn canUseNativeQwenBatch(self: *const TextAdapter, requests: []const task.TaskRequest) bool {
    if (self.family != .qwen3) return false;
    if (requests.len <= 1) return false;

    for (requests, 0..) |request, index| {
        if (!request.generation.native_execution) return false;
        if (request.spec.execution != .sync) return false;
        if (!std.mem.eql(u8, request.spec.operation, "generate") and !std.mem.eql(u8, request.spec.operation, "chat")) return false;
        switch (request.input) {
            .text => {},
            else => return false,
        }

        if (index != 0 and request.generation.max_tokens != requests[0].generation.max_tokens) return false;
    }

    return true;
}

fn canUseNativeQwenSingle(self: *const TextAdapter, request: task.TaskRequest) bool {
    if (self.family != .qwen3) return false;
    if (!request.generation.native_execution) return false;
    if (request.spec.execution != .sync) return false;
    if (!std.mem.eql(u8, request.spec.operation, "generate") and !std.mem.eql(u8, request.spec.operation, "chat")) return false;
    return switch (request.input) {
        .none, .text => true,
        else => false,
    };
}
