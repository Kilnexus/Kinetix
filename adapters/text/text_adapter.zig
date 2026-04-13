const std = @import("std");
const kinetix = @import("engine_root");

const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const adapter_mod = kinetix.adapter;
const registry_mod = kinetix.registry;
const runtime_model = kinetix.runtime.model;
const runtime_providers = kinetix.runtime.providers;
const runtime_session = kinetix.runtime.session;
const task = kinetix.core.task;
const family_registry = kinetix.runtime.text.family_registry;
const text_shared = kinetix.runtime.providers.text_shared;

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
    plan: load_plan.ResolvedLoadPlan,
    runtime_handle: runtime_model.ModelHandle,
    family: ModelFamily,
    adapter_id: []u8,
    descriptor: adapter_mod.Descriptor,

    pub fn init(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        preferred_weights: backend.WeightScheme,
    ) !TextAdapter {
        var catalog = try backend.ModelCatalog.discover(allocator, model_dir);
        defer catalog.deinit();

        const plan = try load_plan.resolve(&catalog, .{
            .model_dir = catalog.model_dir,
            .preferred_weights = preferred_weights,
        });
        errdefer {
            var owned_plan = plan;
            owned_plan.deinit();
        }

        if (plan.config_path == null) return error.MissingConfigArtifact;
        if (plan.tokenizer_path == null) return error.MissingTokenizerArtifact;
        if (plan.weights_path == null) return error.MissingWeightArtifacts;

        const family = try detectModelFamily(allocator, plan.config_path.?);
        const basename = std.fs.path.basename(plan.model_dir);
        const family_name = family.name();
        const adapter_id = try std.fmt.allocPrint(allocator, "text.{s}.{s}", .{ family_name, basename });
        errdefer allocator.free(adapter_id);
        var session = runtime_session.RuntimeSession.init(allocator);
        defer session.deinit();
        const runtime_handle = try session.openModel(.{
            .model_dir = model_dir,
            .preferred_weights = preferred_weights,
        });
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
        self.runtime_handle.deinit();
        self.plan.deinit();
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
            return text_shared.buildReadyResult(try submit(ctx, request));
        }

        return try runtime_providers.adapter_bridge.executeSingle(
            allocator,
            &self.runtime_handle,
            self.descriptor.id,
            request,
        );
    }

    fn executeBatch(ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]adapter_mod.ExecutionResult {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        if (self.family != .qwen3) {
            return try text_shared.buildReadyBatchResults(allocator, self.descriptor.id, requests);
        }

        return try runtime_providers.adapter_bridge.executeBatch(
            allocator,
            &self.runtime_handle,
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

    const architecture = family_registry.detectArchitecture(model_type_value.string) orelse return .unknown;
    return switch (architecture) {
        .qwen3 => .qwen3,
        .bert => .bert,
    };
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
    return text_shared.buildSubmission(self.descriptor.id, request.spec.execution);
}
