const std = @import("std");
const kinetix = @import("root").kinetix;
const fs_compat = @import("engine_fs_compat");

const backend = kinetix.artifacts.backend;
const execution = kinetix.execution;
const task = kinetix.core.task;

pub const Command = union(enum) {
    help,
    run: RunArgs,
    batch_plan: BatchPlanArgs,
    batch_run: BatchPlanArgs,
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
    input: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

pub const BatchPlanArgs = struct {
    model_dir: []const u8,
    requests_file: []const u8,
    operation: ?[]const u8 = null,
    execution: task.ExecutionMode = .sync,
    preferred_weights: backend.WeightScheme = .auto,
    max_tokens: ?usize = null,
    native_exec: bool = false,
};

const BatchRequestJson = struct {
    operation: ?[]const u8 = null,
    input: ?[]const u8 = null,
    execution: ?[]const u8 = null,
    max_tokens: ?usize = null,
    native_exec: ?bool = null,
    allows_batching: ?bool = null,
};

pub fn parse(allocator: std.mem.Allocator, args: []const [:0]const u8) !ParsedCommand {
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
    if (std.mem.eql(u8, cmd, "batch-plan") or std.mem.eql(u8, cmd, "batch-run")) {
        var model_dir: ?[]const u8 = null;
        var requests_file: ?[]const u8 = null;
        var operation: ?[]const u8 = null;
        var execution_mode: task.ExecutionMode = .sync;
        var preferred_weights: backend.WeightScheme = .auto;
        var max_tokens: ?usize = null;
        var native_exec = false;

        var i: usize = 2;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--model-dir")) {
                i += 1;
                if (i >= args.len) return error.MissingModelDir;
                model_dir = copied[i];
                continue;
            }
            if (std.mem.eql(u8, args[i], "--requests-file")) {
                i += 1;
                if (i >= args.len) return error.MissingRequestsFile;
                requests_file = copied[i];
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
                execution_mode = try parseExecutionMode(args[i]);
                continue;
            }
            if (std.mem.eql(u8, args[i], "--weights")) {
                i += 1;
                if (i >= args.len) return error.MissingWeightScheme;
                preferred_weights = try parseWeightScheme(args[i]);
                continue;
            }
            if (std.mem.eql(u8, args[i], "--max-tokens")) {
                i += 1;
                if (i >= args.len) return error.MissingMaxTokens;
                max_tokens = try std.fmt.parseInt(usize, args[i], 10);
                continue;
            }
            if (std.mem.eql(u8, args[i], "--native-exec")) {
                native_exec = true;
                continue;
            }
            return error.UnknownOption;
        }

        if (model_dir == null) return error.MissingModelDir;
        if (requests_file == null) return error.MissingRequestsFile;

        return .{
            .command = if (std.mem.eql(u8, cmd, "batch-run")) .{ .batch_run = .{
                .model_dir = model_dir.?,
                .requests_file = requests_file.?,
                .operation = operation,
                .execution = execution_mode,
                .preferred_weights = preferred_weights,
                .max_tokens = max_tokens,
                .native_exec = native_exec,
            } } else .{ .batch_plan = .{
                .model_dir = model_dir.?,
                .requests_file = requests_file.?,
                .operation = operation,
                .execution = execution_mode,
                .preferred_weights = preferred_weights,
                .max_tokens = max_tokens,
                .native_exec = native_exec,
            } },
            .argv_copy = copied,
        };
    }

    if (!std.mem.eql(u8, cmd, "run")) return error.UnknownCommand;

    var model_dir: ?[]const u8 = null;
    var operation: ?[]const u8 = null;
    var input: ?[]const u8 = null;
    var execution_mode: task.ExecutionMode = .sync;
    var preferred_weights: backend.WeightScheme = .auto;
    var max_tokens: ?usize = null;
    var native_exec = false;

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
        if (std.mem.eql(u8, args[i], "--input")) {
            i += 1;
            if (i >= args.len) return error.MissingInputValue;
            input = copied[i];
            continue;
        }
        if (std.mem.eql(u8, args[i], "--execution")) {
            i += 1;
            if (i >= args.len) return error.MissingExecutionMode;
            execution_mode = try parseExecutionMode(args[i]);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--weights")) {
            i += 1;
            if (i >= args.len) return error.MissingWeightScheme;
            preferred_weights = try parseWeightScheme(args[i]);
            continue;
        }
        if (std.mem.eql(u8, args[i], "--native-exec")) {
            native_exec = true;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingMaxTokens;
            max_tokens = try std.fmt.parseInt(usize, args[i], 10);
            continue;
        }
        return error.UnknownOption;
    }

    if (model_dir == null) return error.MissingModelDir;

    return .{
        .command = .{ .run = .{
            .model_dir = model_dir.?,
            .operation = operation,
            .input = input,
            .execution = execution_mode,
            .preferred_weights = preferred_weights,
            .max_tokens = max_tokens,
            .native_exec = native_exec,
        } },
        .argv_copy = copied,
    };
}

pub fn run(stdout: anytype, stderr: anytype, parsed: ParsedCommand) !void {
    _ = stderr;
    switch (parsed.command) {
        .help => try printUsage(stdout),
        .run => |args| try runCommand(stdout, args),
        .batch_plan => |args| try runBatchPlan(stdout, args),
        .batch_run => |args| try runBatchRun(stdout, args),
    }
}

pub fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Kinetix CLI
        \\
        \\Usage:
        \\  kinetix run --model-dir <path> [--operation <name>] [--input <value>] [--execution sync|async|stream] [--weights auto|bf16|q8|q6|q4] [--native-exec]
        \\  kinetix batch-plan --model-dir <path> --requests-file <json> [--native-exec]
        \\  kinetix batch-run --model-dir <path> --requests-file <json> [--native-exec]
        \\  kinetix --help
        \\
        \\Examples:
        \\  kinetix run --model-dir .\\models\\vision\\compat_yolo11n --operation detect --input .\\datasets\\vision\\archive\\images\\000_0001.png
        \\  kinetix run --model-dir .\\models\\text\\Qwen3-0.6B --operation generate --input "Hello from Kinetix" --max-tokens 8 --native-exec
        \\  kinetix run --model-dir .\\models\\ocr\\PP-OCRv5_server_det_infer --operation infer-ocr --input input.ppm
        \\  kinetix batch-plan --model-dir .\\models\\text\\Qwen3-0.6B --requests-file requests.json
        \\  kinetix batch-run --model-dir .\\models\\text\\Qwen3-0.6B --requests-file requests.json --native-exec
        \\
    );
}

fn runCommand(stdout: anytype, args: RunArgs) !void {
    var prepared = try execution.prepare(std.heap.page_allocator, .{
        .model_dir = args.model_dir,
        .operation = args.operation,
        .input = args.input,
        .execution = args.execution,
        .preferred_weights = args.preferred_weights,
        .max_tokens = args.max_tokens,
        .native_exec = args.native_exec,
    });
    defer prepared.deinit();

    try stdout.print("adapter: {s}\n", .{prepared.descriptor.id});
    try stdout.print("modality: {s}\n", .{@tagName(prepared.descriptor.modality)});
    try stdout.print("model_family: {s}\n", .{prepared.request.spec.model_family});
    try stdout.print("operation: {s}\n", .{prepared.request.spec.operation});
    try stdout.print("execution: {s}\n", .{@tagName(prepared.runtime_plan.execution)});
    try stdout.print("supports_batching: {s}\n", .{boolText(prepared.descriptor.supports_batching)});
    try stdout.print("supports_streaming: {s}\n", .{boolText(prepared.descriptor.supports_streaming)});

    var result = try prepared.execute();
    defer result.deinit(std.heap.page_allocator);
    try stdout.print("accepted: {s}\n", .{boolText(result.submission.accepted)});
    try stdout.print("execution_origin: {s}\n", .{@tagName(result.origin)});
    if (result.note != .none) {
        try stdout.print("execution_note: {s}\n", .{@tagName(result.note)});
    }
    switch (result.output) {
        .none => {},
        .text => |value| try stdout.print("output_text: {s}\n", .{value}),
        .json => |value| try stdout.print("output_json: {s}\n", .{value}),
    }
}

fn runBatchPlan(stdout: anytype, args: BatchPlanArgs) !void {
    const items = try loadBatchItems(args);
    defer freeBatchItems(items);

    var prepared = try execution.prepareBatch(std.heap.page_allocator, .{
        .model_dir = args.model_dir,
        .preferred_weights = args.preferred_weights,
        .items = items,
    });
    defer prepared.deinit();

    try stdout.print("adapter: {s}\n", .{prepared.descriptor.id});
    try stdout.print("modality: {s}\n", .{@tagName(prepared.descriptor.modality)});
    try stdout.print("model_family: {s}\n", .{prepared.descriptor.bound_model_family.?});
    try stdout.print("requests: {d}\n", .{prepared.requests.len});
    try stdout.print("batches: {d}\n", .{prepared.runtime_plan.batches.len});

    for (prepared.runtime_plan.batches, 0..) |batch, batch_index| {
        try stdout.print("batch[{d}]: size={d} execution={s} batching={s} operation={s}", .{
            batch_index,
            batch.request_indices.len,
            @tagName(batch.execution),
            boolText(batch.allows_batching),
            batch.operation,
        });
        try stdout.writeAll(" indices=");
        for (batch.request_indices, 0..) |request_index, request_offset| {
            if (request_offset != 0) try stdout.writeAll(",");
            try stdout.print("{d}", .{request_index});
        }
        try stdout.writeAll("\n");
    }
}

fn runBatchRun(stdout: anytype, args: BatchPlanArgs) !void {
    const items = try loadBatchItems(args);
    defer freeBatchItems(items);

    var prepared = try execution.prepareBatch(std.heap.page_allocator, .{
        .model_dir = args.model_dir,
        .preferred_weights = args.preferred_weights,
        .items = items,
    });
    defer prepared.deinit();

    var report = try prepared.execute();
    defer report.deinit();

    try stdout.print("adapter: {s}\n", .{prepared.descriptor.id});
    try stdout.print("modality: {s}\n", .{@tagName(prepared.descriptor.modality)});
    try stdout.print("model_family: {s}\n", .{prepared.descriptor.bound_model_family.?});
    try stdout.print("requests: {d}\n", .{report.totalRequests()});
    try stdout.print("accepted: {d}\n", .{report.totalAccepted()});
    try stdout.print("batches: {d}\n", .{report.batches.len});

    for (report.batches, 0..) |batch, batch_index| {
        try stdout.print("batch[{d}]: size={d} accepted={d} execution={s} batching={s} path={s}", .{
            batch_index,
            batch.len(),
            batch.acceptedCount(),
            @tagName(batch.execution),
            boolText(batch.supports_batching),
            @tagName(batch.execute_path),
        });
        try stdout.writeAll(" indices=");
        for (batch.request_results, 0..) |result, result_index| {
            if (result_index != 0) try stdout.writeAll(",");
            try stdout.print("{d}", .{result.request_index});
        }
        try stdout.writeAll("\n");

        for (batch.request_results) |result| {
            switch (result.result.output) {
                .none => {},
                .text => |value| try stdout.print("text[{d}]: {s}\n", .{ result.request_index, value }),
                .json => |value| try stdout.print("json[{d}]: {s}\n", .{ result.request_index, value }),
            }
        }
    }
}

fn loadBatchItems(args: BatchPlanArgs) ![]execution.PrepareBatchItem {
    const file_bytes = try fs_compat.cwd().readFileAlloc(std.heap.page_allocator, args.requests_file, 4 * 1024 * 1024);
    defer std.heap.page_allocator.free(file_bytes);

    const parsed = try std.json.parseFromSlice([]BatchRequestJson, std.heap.page_allocator, file_bytes, .{});
    defer parsed.deinit();

    const items = try std.heap.page_allocator.alloc(execution.PrepareBatchItem, parsed.value.len);
    errdefer {
        for (items, 0..) |item, index| {
            if (index >= parsed.value.len) break;
            freeBatchItem(item);
        }
        std.heap.page_allocator.free(items);
    }

    for (parsed.value, items) |entry, *item| {
        item.* = .{
            .operation = if (entry.operation) |operation|
                try std.heap.page_allocator.dupe(u8, operation)
            else if (args.operation) |operation|
                try std.heap.page_allocator.dupe(u8, operation)
            else
                null,
            .input = if (entry.input) |input|
                try std.heap.page_allocator.dupe(u8, input)
            else
                null,
            .execution = if (entry.execution) |mode| try parseExecutionMode(mode) else args.execution,
            .max_tokens = entry.max_tokens orelse args.max_tokens,
            .native_exec = entry.native_exec orelse args.native_exec,
            .allows_batching = entry.allows_batching orelse true,
        };
    }
    return items;
}

fn freeBatchItems(items: []execution.PrepareBatchItem) void {
    for (items) |item| freeBatchItem(item);
    std.heap.page_allocator.free(items);
}

fn freeBatchItem(item: execution.PrepareBatchItem) void {
    if (item.operation) |operation| std.heap.page_allocator.free(operation);
    if (item.input) |input| std.heap.page_allocator.free(input);
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
