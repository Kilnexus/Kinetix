const std = @import("std");
const task = @import("../core/task.zig");

pub const Descriptor = struct {
    id: []const u8,
    modality: task.Modality,
    version: []const u8 = "0.1.0",
    bound_model_family: ?[]const u8 = null,
    supports_batching: bool = false,
    supports_streaming: bool = false,
    supported_operations: []const []const u8 = &.{},

    pub fn supportsOperation(self: Descriptor, operation: []const u8) bool {
        if (self.supported_operations.len == 0) return true;
        for (self.supported_operations) |supported| {
            if (std.mem.eql(u8, supported, operation)) return true;
        }
        return false;
    }

    pub fn supportsModelFamily(self: Descriptor, model_family: []const u8) bool {
        if (self.bound_model_family) |bound| {
            return std.mem.eql(u8, bound, model_family);
        }
        return true;
    }
};

pub const Submission = struct {
    adapter_id: []const u8,
    accepted: bool = true,
    execution: task.ExecutionMode,
};

pub const OutputPayload = union(enum) {
    none,
    text: []const u8,
    json: []const u8,

    pub fn deinit(self: *OutputPayload, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .none => {},
            .text => |value| allocator.free(value),
            .json => |value| allocator.free(value),
        }
        self.* = .none;
    }
};

pub const ExecutionOrigin = enum {
    shared_adapter,
    legacy_process_bridge,
    native_single_bridge,
    native_batch_bridge,
};

pub const ExecutionNote = enum {
    none,
    validated_only,
    text_request_ready,
    text_native_qwen_single,
    text_native_qwen_batch,
    vision_graph_ready,
    vision_legacy_detect_json,
    ocr_model_ready,
    ocr_legacy_infer_summary,
};

pub const ExecutionResult = struct {
    submission: Submission,
    origin: ExecutionOrigin = .shared_adapter,
    note: ExecutionNote = .none,
    output: OutputPayload = .none,

    pub fn deinit(self: *ExecutionResult, allocator: std.mem.Allocator) void {
        self.output.deinit(allocator);
        self.* = undefined;
    }
};

pub const BatchExecutionPath = enum {
    adapter_batch,
    per_request_fallback,
};

pub const VTable = struct {
    submit: *const fn (ctx: *anyopaque, request: task.TaskRequest) anyerror!Submission,
    submit_batch: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) anyerror![]Submission = null,
    execute: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator, request: task.TaskRequest) anyerror!ExecutionResult = null,
    execute_batch: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) anyerror![]ExecutionResult = null,
};

pub const Adapter = struct {
    ctx: *anyopaque,
    descriptor: Descriptor,
    vtable: *const VTable,

    pub fn submit(self: Adapter, request: task.TaskRequest) !Submission {
        try self.validateRequest(request);
        return try self.vtable.submit(self.ctx, request);
    }

    pub fn submitBatch(self: Adapter, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]Submission {
        for (requests) |request| try self.validateRequest(request);

        if (self.batchSubmitPath(requests.len) == .adapter_batch) {
            return try self.vtable.submit_batch.?(self.ctx, allocator, requests);
        }

        const submissions = try allocator.alloc(Submission, requests.len);
        errdefer allocator.free(submissions);

        for (requests, submissions) |request, *submission| {
            submission.* = try self.vtable.submit(self.ctx, request);
        }

        return submissions;
    }

    pub fn execute(self: Adapter, allocator: std.mem.Allocator, request: task.TaskRequest) !ExecutionResult {
        try self.validateRequest(request);

        if (self.vtable.execute) |execute_fn| {
            return try execute_fn(self.ctx, allocator, request);
        }

        return .{
            .submission = try self.vtable.submit(self.ctx, request),
            .origin = .shared_adapter,
            .note = .validated_only,
        };
    }

    pub fn executeBatch(self: Adapter, allocator: std.mem.Allocator, requests: []const task.TaskRequest) ![]ExecutionResult {
        for (requests) |request| try self.validateRequest(request);

        if (self.batchExecutePath(requests.len) == .adapter_batch) {
            return try self.vtable.execute_batch.?(self.ctx, allocator, requests);
        }

        const results = try allocator.alloc(ExecutionResult, requests.len);
        errdefer allocator.free(results);

        for (requests, results) |request, *result| {
            result.* = try self.execute(allocator, request);
        }

        return results;
    }

    pub fn batchSubmitPath(self: Adapter, request_count: usize) BatchExecutionPath {
        if (request_count > 1 and self.vtable.submit_batch != null) return .adapter_batch;
        return .per_request_fallback;
    }

    pub fn batchExecutePath(self: Adapter, request_count: usize) BatchExecutionPath {
        if (request_count > 1 and self.vtable.execute_batch != null) return .adapter_batch;
        return .per_request_fallback;
    }

    fn validateRequest(self: Adapter, request: task.TaskRequest) !void {
        const spec = request.spec;
        if (spec.modality != self.descriptor.modality) return error.ModalityMismatch;
        if (!self.descriptor.supportsModelFamily(spec.model_family)) return error.ModelFamilyMismatch;
        if (!self.descriptor.supportsOperation(spec.operation)) return error.OperationNotSupported;
        if (spec.execution == .stream and !self.descriptor.supports_streaming) return error.StreamingNotSupported;
    }
};
