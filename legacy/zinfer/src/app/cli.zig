const std = @import("std");
const cli_args = @import("cli/args.zig");
const cli_chat = @import("cli/chat.zig");
const cli_generate = @import("cli/generate.zig");
const cli_embed_text = @import("cli/embed_text.zig");
const cli_fill_mask = @import("cli/fill_mask.zig");
const cli_serve_bert = @import("cli/serve_bert.zig");
const cli_tools = @import("cli/tools.zig");
const cli_usage = @import("cli/usage.zig");
const bert_mlm = @import("../model/runtime/bert_mlm.zig");

const default_model_dir = cli_args.default_model_dir;
const default_bert_model_dir = "models/bert-base-uncased";

pub fn run(allocator: std.mem.Allocator) !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len == 1) {
        try printUsage();
        return error.InvalidCommand;
    }

    const command = args[1];
    if (std.mem.eql(u8, command, "quantize")) {
        if (args.len == 3) {
            try cli_tools.quantizeModelDir(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.quantizeModelDir(allocator, args[3], args[2]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "tokenize")) {
        if (args.len == 3) {
            try cli_tools.tokenizeText(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.tokenizeText(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "decode-ids")) {
        if (args.len == 3) {
            try cli_tools.decodeIds(allocator, default_model_dir, args[2]);
            return;
        }
        if (args.len >= 4) {
            try cli_tools.decodeIds(allocator, args[2], args[3]);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "fill-mask")) {
        if (args.len == 3) {
            try cli_fill_mask.fillMaskText(allocator, default_bert_model_dir, args[2], 5);
            return;
        }
        if (args.len == 4) {
            if (std.fmt.parseInt(usize, args[3], 10)) |top_k| {
                try cli_fill_mask.fillMaskText(allocator, default_bert_model_dir, args[2], top_k);
                return;
            } else |_| {
                try cli_fill_mask.fillMaskText(allocator, args[2], args[3], 5);
                return;
            }
        }
        if (args.len >= 5) {
            const top_k = try std.fmt.parseInt(usize, args[4], 10);
            try cli_fill_mask.fillMaskText(allocator, args[2], args[3], top_k);
            return;
        }
        try printUsage();
        return error.InvalidCommand;
    }

    if (std.mem.eql(u8, command, "embed-text")) {
        var model_dir: []const u8 = default_bert_model_dir;
        var text: []const u8 = undefined;
        var mode: bert_mlm.EmbeddingMode = .mean;
        var count: usize = 16;

        var index: usize = 2;
        if (args.len > index + 1 and looksLikePath(args[index])) {
            model_dir = args[index];
            index += 1;
        }
        if (args.len <= index) {
            try printUsage();
            return error.InvalidCommand;
        }
        text = args[index];
        index += 1;

        if (args.len > index) {
            if (parseEmbeddingMode(args[index])) |parsed_mode| {
                mode = parsed_mode;
                index += 1;
            } else |_| {}
        }
        if (args.len > index) {
            count = try std.fmt.parseInt(usize, args[index], 10);
            index += 1;
        }
        if (args.len != index) {
            try printUsage();
            return error.InvalidCommand;
        }

        try cli_embed_text.embedText(allocator, model_dir, text, mode, count);
        return;
    }

    if (std.mem.eql(u8, command, "serve-bert")) {
        var model_dir: []const u8 = default_bert_model_dir;
        var bind_host: []const u8 = "0.0.0.0";
        var port: u16 = 8787;
        var runtime_count: usize = defaultServeRuntimeCount();

        if (args.len >= 3 and looksLikePath(args[2])) {
            model_dir = args[2];
            if (args.len >= 4) {
                port = try std.fmt.parseInt(u16, args[3], 10);
            }
            if (args.len >= 5) {
                bind_host = args[4];
            }
            if (args.len >= 6) {
                runtime_count = try std.fmt.parseInt(usize, args[5], 10);
            }
        } else {
            if (args.len >= 3) {
                port = try std.fmt.parseInt(u16, args[2], 10);
            }
            if (args.len >= 4) {
                bind_host = args[3];
            }
            if (args.len >= 5) {
                runtime_count = try std.fmt.parseInt(usize, args[4], 10);
            }
        }

        try cli_serve_bert.serve(allocator, model_dir, bind_host, port, runtime_count);
        return;
    }

    if (std.mem.eql(u8, command, "generate")) {
        var invocation = try cli_args.parseGenerateInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_generate.generateText(allocator, invocation.model_dir, invocation.user_text, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "generate-chat")) {
        var invocation = try cli_args.parseGenerateChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_generate.generateChatFromFile(allocator, invocation.model_dir, invocation.messages_json_path, invocation.options);
        return;
    }

    if (std.mem.eql(u8, command, "chat")) {
        var invocation = try cli_args.parseChatInvocation(allocator, args);
        defer invocation.deinit(allocator);
        try cli_chat.chatLoop(
            allocator,
            invocation.model_dir,
            invocation.options,
            invocation.load_path,
            invocation.save_path,
        );
        return;
    }

    if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        try printUsage();
        return;
    }

    std.log.err("unknown command: {s}", .{command});
    try printUsage();
    return error.InvalidCommand;
}

fn printUsage() !void {
    try cli_usage.printUsage();
}

fn looksLikePath(arg: []const u8) bool {
    return std.mem.indexOfScalar(u8, arg, '\\') != null or
        std.mem.indexOfScalar(u8, arg, '/') != null or
        std.mem.startsWith(u8, arg, ".");
}

fn parseEmbeddingMode(text: []const u8) !bert_mlm.EmbeddingMode {
    if (std.mem.eql(u8, text, "cls")) return .cls;
    if (std.mem.eql(u8, text, "mean")) return .mean;
    return error.InvalidEmbeddingMode;
}

fn defaultServeRuntimeCount() usize {
    const cpu_count = @max(@as(usize, 1), std.Thread.getCpuCount() catch 1);
    return @max(@as(usize, 1), cpu_count / 8);
}
