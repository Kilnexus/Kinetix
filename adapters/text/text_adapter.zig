const std = @import("std");
const builtin = @import("builtin");
const kinetix = @import("../../engine/kinetix.zig");

const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const adapter_mod = kinetix.adapter;
const registry_mod = kinetix.registry;
const task = kinetix.core.task;
const text_native_dispatch = kinetix.runtime.text.native_dispatch;
const native_batch_bridge = if (builtin.is_test) struct {
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
} else text_native_dispatch.NativeBatchBridge;

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
const unknown_operations = [_][]const u8{"infer-text"};

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
        if (self.family != .qwen3) {
            return .{
                .submission = try submit(ctx, request),
                .origin = .shared_adapter,
                .note = .text_request_ready,
            };
        }

        return try text_native_dispatch.executeSingle(
            allocator,
            native_batch_bridge,
            self.catalog.model_dir,
            self.plan.weight_scheme orelse .auto,
            try submit(ctx, request),
            request,
        );
    }

    fn executeBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.ExecutionResult {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        if (self.family != .qwen3) {
            const results = try allocator.alloc(adapter_mod.ExecutionResult, requests.len);
            errdefer allocator.free(results);

            for (requests, results) |request, *result| {
                result.* = .{
                    .submission = buildSubmission(self, request),
                    .origin = .shared_adapter,
                    .note = .text_request_ready,
                };
            }
            return results;
        }

        return try text_native_dispatch.executeBatch(
            allocator,
            native_batch_bridge,
            self.catalog.model_dir,
            self.plan.weight_scheme orelse .auto,
            self.descriptor.id,
            requests,
        );
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
