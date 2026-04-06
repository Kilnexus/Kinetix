const std = @import("std");
const kinetix = @import("../../kinetix.zig");

const backend = kinetix.artifacts.backend;
const load_plan = kinetix.runtime.load_plan;
const adapter_mod = kinetix.adapter;
const legacy_command = kinetix.adapters.legacy_command;
const registry_mod = kinetix.registry;
const task = kinetix.core.task;

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

    fn submit(ctx: *anyopaque, spec: task.TaskSpec) !adapter_mod.Submission {
        const self: *TextAdapter = @ptrCast(@alignCast(ctx));
        if (self.plan.weights_path == null) return error.MissingWeightArtifacts;
        if (self.plan.config_path == null) return error.MissingConfigArtifact;
        if (self.plan.tokenizer_path == null) return error.MissingTokenizerArtifact;

        return .{
            .adapter_id = self.descriptor.id,
            .accepted = true,
            .execution = spec.execution,
        };
    }
};

const vtable = adapter_mod.VTable{
    .submit = TextAdapter.submit,
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
