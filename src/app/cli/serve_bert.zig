const std = @import("std");
const bert_mlm = @import("../../model/runtime/bert_mlm.zig");

const http = std.http;

const json_header = [_]http.Header{
    .{ .name = "content-type", .value = "application/json" },
};

const FillMaskRequest = struct {
    text: []const u8,
    top_k: ?usize = null,
};

const EmbedRequest = struct {
    text: []const u8,
    mode: ?[]const u8 = null,
    count: ?usize = null,
};

pub fn serve(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    bind_host: []const u8,
    port: u16,
) !void {
    var runtime = try bert_mlm.Runtime.init(allocator, model_dir);
    defer runtime.deinit();

    const address = try std.net.Address.parseIp(bind_host, port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer serve-bert\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("bind: {f}\n", .{address});
    try stdout.print("backend: {s}\n", .{runtime.backend.resolvedScheme().name()});
    try stdout.print("threads: {d}\n", .{runtime.thread_count});

    while (true) {
        const connection = try server.accept();
        handleConnection(allocator, &runtime, connection) catch |err| {
            const stderr = std.fs.File.stderr().deprecatedWriter();
            stderr.print("serve-bert connection error: {t}\n", .{err}) catch {};
        };
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    runtime: *bert_mlm.Runtime,
    connection: std.net.Server.Connection,
) !void {
    defer connection.stream.close();

    var send_buffer: [4096]u8 = undefined;
    var recv_buffer: [4096]u8 = undefined;
    var body_buffer: [4096]u8 = undefined;

    var connection_reader = connection.stream.reader(&recv_buffer);
    var connection_writer = connection.stream.writer(&send_buffer);
    var server: http.Server = .init(connection_reader.interface(), &connection_writer.interface);

    while (true) {
        var request = server.receiveHead() catch |err| switch (err) {
            error.HttpConnectionClosing => return,
            else => return err,
        };
        try serveRequest(allocator, runtime, &request, &body_buffer);
    }
}

fn serveRequest(
    allocator: std.mem.Allocator,
    runtime: *bert_mlm.Runtime,
    request: *http.Server.Request,
    body_buffer: []u8,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    if (request.head.method == .GET and std.mem.eql(u8, request.head.target, "/health")) {
        var response = std.ArrayListUnmanaged(u8).empty;
        defer response.deinit(arena);
        try response.writer(arena).print("{f}", .{std.json.fmt(.{
            .status = "ok",
            .backend = runtime.backend.resolvedScheme().name(),
            .threads = runtime.thread_count,
        }, .{})});
        try request.respond(response.items, .{ .extra_headers = &json_header });
        return;
    }

    if (request.head.method == .POST and std.mem.eql(u8, request.head.target, "/fill-mask")) {
        const body = try readJsonBody(arena, request, body_buffer);
        const parsed = try std.json.parseFromSliceLeaky(FillMaskRequest, arena, body, .{
            .ignore_unknown_fields = true,
        });

        var timer = try std.time.Timer.start();
        var result = try runtime.fillMask(parsed.text, parsed.top_k orelse 5);
        defer result.deinit(allocator);

        var response = std.ArrayListUnmanaged(u8).empty;
        defer response.deinit(arena);
        try response.writer(arena).print("{f}", .{std.json.fmt(.{
            .mask_position = result.mask_position,
            .predictions = result.predictions,
            .elapsed_ms = nsToMs(timer.read()),
        }, .{})});
        try request.respond(response.items, .{ .extra_headers = &json_header });
        return;
    }

    if (request.head.method == .POST and std.mem.eql(u8, request.head.target, "/embed")) {
        const body = try readJsonBody(arena, request, body_buffer);
        const parsed = try std.json.parseFromSliceLeaky(EmbedRequest, arena, body, .{
            .ignore_unknown_fields = true,
        });
        const mode = if (parsed.mode) |mode_text|
            parseEmbeddingMode(mode_text)
        else
            bert_mlm.EmbeddingMode.mean;

        var timer = try std.time.Timer.start();
        const embedding = try runtime.embedText(parsed.text, mode);
        defer allocator.free(embedding);

        const vector = embedding[0 .. parsed.count orelse embedding.len];
        var response = std.ArrayListUnmanaged(u8).empty;
        defer response.deinit(arena);
        try response.writer(arena).print("{f}", .{std.json.fmt(.{
            .mode = mode.name(),
            .dims = embedding.len,
            .vector = vector,
            .elapsed_ms = nsToMs(timer.read()),
        }, .{})});
        try request.respond(response.items, .{ .extra_headers = &json_header });
        return;
    }

    if (request.head.method != .GET and request.head.method != .POST) {
        try request.respond("{\"error\":\"method not allowed\"}", .{
            .status = .method_not_allowed,
            .extra_headers = &json_header,
        });
        return;
    }

    try request.respond("{\"error\":\"not found\"}", .{
        .status = .not_found,
        .extra_headers = &json_header,
    });
}

fn readJsonBody(
    allocator: std.mem.Allocator,
    request: *http.Server.Request,
    body_buffer: []u8,
) ![]u8 {
    const content_length = request.head.content_length orelse return error.MissingContentLength;
    const reader = try request.readerExpectContinue(body_buffer);
    const body = try allocator.alloc(u8, content_length);
    try reader.readSliceAll(body);
    return body;
}

fn parseEmbeddingMode(text: []const u8) bert_mlm.EmbeddingMode {
    if (std.mem.eql(u8, text, "cls")) return .cls;
    return .mean;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}
