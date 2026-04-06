const std = @import("std");
const kinetix = @import("../../kinetix.zig");

const factory = kinetix.adapters.factory;
const backend = kinetix.artifacts.backend;
const registry_mod = kinetix.registry;
const scheduler_mod = kinetix.scheduler;
const task = kinetix.core.task;

pub const Command = union(enum) {
    help,
    run: RunArgs,
};

pub const ParsedCommand = struct {
    command: Command,
    argv_copy: [][]u8,

    pub fn deinit(self: ParsedCommand, allocator: std.mem.Allocator) void {
        for (self.argv_copy) |arg| allocator.free(arg);
        allocator.free(self.argv_copy);
    }
};

pub const RunArgs = struct {
    model_dir: []const u8,
    operation: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
};

pub fn parse(allocator: std.mem.Allocator) !ParsedCommand {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var copied = try allocator.alloc([]u8, args.len);
    errdefer {
        for (copied, 0..) |arg, idx| {
            if (idx >= args.len) break;
            allocator.free(arg);
        }
        allocator.free(copied);
    }
    for (args, 0..) |arg, idx| {
        copied[idx] = try allocator.dupe(u8, arg);
    }

    if (args.len <= 1) {
        return .{ .command = .help, .argv_copy = copied };
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h") or std.mem.eql(u8, cmd, "help")) {
        return .{ .command = .help, .argv_copy = copied };
    }
    if (!std.mem.eql(u8, cmd, "run")) return error.UnknownCommand;

    var model_dir: ?[]const u8 = null;
    var operation: ?[]const u8 = null;
    var execution: task.ExecutionMode = .sync;
    var preferred_weights: backend.WeightScheme = .auto;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model-dir")) {
            i += 1;
            if (i >= args.len) return error.MissingModelDir;
            model_dir = copied[i];
            continue;
        }
        if (std.mem.eql(u8, args[i], "--operation")) {
            i += 1;
            if (i >= args.len) return error.MissingOperation;
            operation = copied[i];
            continue;
        }
        if (std.mem.eql(u8, args[i], "--execution")) {
            i += 1;
            if (i >= args.len) return error.MissingExecutionMode;
            execution = try parseExecutionMode(args[i]);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--weights")) {
            i += 1;
            if (i >= args.len) return error.MissingWeightScheme;
            preferred_weights = try parseWeightScheme(args[i]);
            continue;
        }
        return error.UnknownOption;
    }

    if (model_dir == null) return error.MissingModelDir;

    return .{
        .command = .{ .run = .{
            .model_dir = model_dir.?,
            .operation = operation,
            .execution = execution,
            .preferred_weights = preferred_weights,
        } },
        .argv_copy = copied,
    };
}

pub fn run(stdout: anytype, stderr: anytype, parsed: ParsedCommand) !void {
    _ = stderr;
    switch (parsed.command) {
        .help => try printUsage(stdout),
        .run => |args| try runCommand(stdout, args),
    }
}

pub fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Kinetix CLI
        \\
        \\Usage:
        \\  kinetix run --model-dir <path> [--operation <name>] [--execution sync|async|stream] [--weights auto|bf16|q8|q6|q4]
        \\  kinetix --help
        \\
        \\Examples:
        \\  kinetix run --model-dir .\\legacy\\axionyx\\artifacts --operation detect
        \\  kinetix run --model-dir .\\models\\Qwen3-0.6B --execution stream
        \\  kinetix run --model-dir .\\ocr-demo --operation infer-ocr
        \\
    );
}

fn runCommand(stdout: anytype, args: RunArgs) !void {
    var registry = registry_mod.Registry.init(std.heap.page_allocator);
    defer registry.deinit();

    var managed = try factory.initAuto(std.heap.page_allocator, args.model_dir, args.preferred_weights);
    defer managed.deinit();
    try managed.registerInto(&registry);

    const descriptor = managed.descriptor();
    const operation = args.operation orelse defaultOperation(descriptor);
    const model_family = descriptor.bound_model_family orelse return error.MissingModelFamilyBinding;
    const scheduler = scheduler_mod.Scheduler.init(&registry);

    const spec = task.TaskSpec{
        .modality = descriptor.modality,
        .operation = operation,
        .model_family = model_family,
        .adapter_id = descriptor.id,
        .execution = args.execution,
    };

    const plan = try scheduler.plan(spec);
    const submission = try scheduler.submit(spec);

    try stdout.print("adapter: {s}\n", .{descriptor.id});
    try stdout.print("modality: {s}\n", .{@tagName(descriptor.modality)});
    try stdout.print("model_family: {s}\n", .{model_family});
    try stdout.print("operation: {s}\n", .{operation});
    try stdout.print("execution: {s}\n", .{@tagName(plan.execution)});
    try stdout.print("supports_batching: {s}\n", .{boolText(plan.supports_batching)});
    try stdout.print("supports_streaming: {s}\n", .{boolText(plan.supports_streaming)});
    try stdout.print("accepted: {s}\n", .{boolText(submission.accepted)});
}

fn defaultOperation(descriptor: kinetix.adapter.Descriptor) []const u8 {
    if (descriptor.supported_operations.len == 0) return "infer";
    return descriptor.supported_operations[0];
}

fn boolText(value: bool) []const u8 {
    return if (value) "true" else "false";
}

fn parseExecutionMode(value: []const u8) !task.ExecutionMode {
    if (std.mem.eql(u8, value, "sync")) return .sync;
    if (std.mem.eql(u8, value, "async")) return .async;
    if (std.mem.eql(u8, value, "stream")) return .stream;
    return error.InvalidExecutionMode;
}

fn parseWeightScheme(value: []const u8) !backend.WeightScheme {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "bf16")) return .bf16;
    if (std.mem.eql(u8, value, "q8")) return .q8;
    if (std.mem.eql(u8, value, "q6")) return .q6;
    if (std.mem.eql(u8, value, "q4")) return .q4;
    return error.InvalidWeightScheme;
}
