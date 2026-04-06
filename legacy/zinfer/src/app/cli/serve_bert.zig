const std = @import("std");
const bert_mlm = @import("../../model/runtime/bert_mlm.zig");

const http = std.http;

const json_header = [_]http.Header{
    .{ .name = "content-type", .value = "application/json" },
};

const service_allocator = std.heap.page_allocator;

const FillMaskRequest = struct {
    text: []const u8,
    top_k: ?usize = null,
};

const EmbedRequest = struct {
    text: []const u8,
    mode: ?[]const u8 = null,
    count: ?usize = null,
};

const RuntimeLease = struct {
    pool: *RuntimePool,
    index: usize,
    runtime: *bert_mlm.Runtime,

    fn release(self: RuntimeLease) void {
        self.pool.release(self.index);
    }
};

const RuntimePool = struct {
    allocator: std.mem.Allocator,
    runtimes: []bert_mlm.Runtime,
    available: []bool,
    thread_count_per_runtime: usize,
    mutex: std.Thread.Mutex = .{},
    condition: std.Thread.Condition = .{},

    fn init(allocator: std.mem.Allocator, model_dir: []const u8, runtime_count: usize) !RuntimePool {
        const actual_runtime_count = @max(@as(usize, 1), runtime_count);
        const cpu_count = @max(@as(usize, 1), std.Thread.getCpuCount() catch 1);
        const threads_per_runtime = @max(@as(usize, 1), cpu_count / actual_runtime_count);

        const runtimes = try allocator.alloc(bert_mlm.Runtime, actual_runtime_count);
        errdefer allocator.free(runtimes);
        const available = try allocator.alloc(bool, actual_runtime_count);
        errdefer allocator.free(available);

        var initialized: usize = 0;
        errdefer {
            for (runtimes[0..initialized]) |*runtime| runtime.deinit();
        }

        for (runtimes, 0..) |*runtime, idx| {
            runtime.* = try bert_mlm.Runtime.initWithThreads(allocator, model_dir, threads_per_runtime);
            available[idx] = true;
            initialized += 1;
        }

        return .{
            .allocator = allocator,
            .runtimes = runtimes,
            .available = available,
            .thread_count_per_runtime = threads_per_runtime,
        };
    }

    fn deinit(self: *RuntimePool) void {
        for (self.runtimes) |*runtime| runtime.deinit();
        self.allocator.free(self.available);
        self.allocator.free(self.runtimes);
    }

    fn acquire(self: *RuntimePool) RuntimeLease {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (true) {
            for (self.available, 0..) |is_available, idx| {
                if (!is_available) continue;
                self.available[idx] = false;
                return .{
                    .pool = self,
                    .index = idx,
                    .runtime = &self.runtimes[idx],
                };
            }
            self.condition.wait(&self.mutex);
        }
    }

    fn release(self: *RuntimePool, index: usize) void {
        self.mutex.lock();
        self.available[index] = true;
        self.mutex.unlock();
        self.condition.signal();
    }

    fn availableCount(self: *RuntimePool) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        var count: usize = 0;
        for (self.available) |is_available| {
            count += @intFromBool(is_available);
        }
        return count;
    }
};

const ServerContext = struct {
    pool: RuntimePool,

    fn deinit(self: *ServerContext) void {
        self.pool.deinit();
    }
};

const ConnectionContext = struct {
    server: *ServerContext,
    connection: std.net.Server.Connection,
};

pub fn serve(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    bind_host: []const u8,
    port: u16,
    runtime_count: usize,
) !void {
    _ = allocator;

    var server_ctx = ServerContext{
        .pool = try RuntimePool.init(service_allocator, model_dir, runtime_count),
    };
    defer server_ctx.deinit();

    const address = try std.net.Address.parseIp(bind_host, port);
    var listener = try address.listen(.{ .reuse_address = true });
    defer listener.deinit();

    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("Zinfer serve-bert\n", .{});
    try stdout.print("model_dir: {s}\n", .{model_dir});
    try stdout.print("bind: {f}\n", .{address});
    try stdout.print("backend: {s}\n", .{server_ctx.pool.runtimes[0].backend.resolvedScheme().name()});
    try stdout.print("runtime_count: {d}\n", .{server_ctx.pool.runtimes.len});
    try stdout.print("threads_per_runtime: {d}\n", .{server_ctx.pool.thread_count_per_runtime});

    while (true) {
        const connection = try listener.accept();
        const context = try service_allocator.create(ConnectionContext);
        context.* = .{
            .server = &server_ctx,
            .connection = connection,
        };

        const thread = try std.Thread.spawn(.{}, connectionMain, .{context});
        thread.detach();
    }
}

fn connectionMain(context: *ConnectionContext) void {
    defer service_allocator.destroy(context);

    handleConnection(context.server, context.connection) catch |err| {
        const stderr = std.fs.File.stderr().deprecatedWriter();
        stderr.print("serve-bert connection error: {t}\n", .{err}) catch {};
    };
}

fn handleConnection(server_ctx: *ServerContext, connection: std.net.Server.Connection) !void {
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
        try serveRequest(server_ctx, &request, &body_buffer);
    }
}

fn serveRequest(
    server_ctx: *ServerContext,
    request: *http.Server.Request,
    body_buffer: []u8,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(service_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    if (request.head.method == .GET and std.mem.eql(u8, request.head.target, "/health")) {
        var response = std.ArrayListUnmanaged(u8).empty;
        defer response.deinit(arena);
        try response.writer(arena).print("{f}", .{std.json.fmt(.{
            .status = "ok",
            .backend = server_ctx.pool.runtimes[0].backend.resolvedScheme().name(),
            .runtime_count = server_ctx.pool.runtimes.len,
            .available_runtimes = server_ctx.pool.availableCount(),
            .threads_per_runtime = server_ctx.pool.thread_count_per_runtime,
        }, .{})});
        try request.respond(response.items, .{ .extra_headers = &json_header });
        return;
    }

    if (request.head.method == .POST and std.mem.eql(u8, request.head.target, "/fill-mask")) {
        const body = try readJsonBody(arena, request, body_buffer);
        const parsed = try std.json.parseFromSliceLeaky(FillMaskRequest, arena, body, .{
            .ignore_unknown_fields = true,
        });

        const lease = server_ctx.pool.acquire();
        defer lease.release();

        var timer = try std.time.Timer.start();
        var result = try lease.runtime.fillMask(parsed.text, parsed.top_k orelse 5);
        defer result.deinit(lease.runtime.allocator);

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

        const lease = server_ctx.pool.acquire();
        defer lease.release();

        var timer = try std.time.Timer.start();
        const embedding = try lease.runtime.embedText(parsed.text, mode);
        defer lease.runtime.allocator.free(embedding);

        const vector = embedding[0..@min(parsed.count orelse embedding.len, embedding.len)];
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
