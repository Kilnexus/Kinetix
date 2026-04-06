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

pub const BatchSubmitPath = enum {
    adapter_batch,
    per_request_fallback,
};

pub const VTable = struct {
    submit: *const fn (ctx: *anyopaque, request: task.TaskRequest) anyerror!Submission,
    submit_batch: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator, requests: []const task.TaskRequest) anyerror![]Submission = null,
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

    pub fn batchSubmitPath(self: Adapter, request_count: usize) BatchSubmitPath {
        if (request_count > 1 and self.vtable.submit_batch != null) return .adapter_batch;
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
