const std = @import("std");
const task = @import("../core/task.zig");
const adapter_mod = @import("../adapter/adapter.zig");

pub const Entry = struct {
    adapter: adapter_mod.Adapter,
};

pub const Registry = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(Entry) = .empty,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Registry) void {
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn register(self: *Registry, adapter: adapter_mod.Adapter) !void {
        if (self.findById(adapter.descriptor.id) != null) return error.AdapterAlreadyRegistered;
        try self.entries.append(self.allocator, .{
            .adapter = adapter,
        });
    }

    pub fn count(self: *const Registry) usize {
        return self.entries.items.len;
    }

    pub fn findById(self: *const Registry, id: []const u8) ?*const Entry {
        for (self.entries.items) |*entry| {
            if (std.mem.eql(u8, entry.adapter.descriptor.id, id)) return entry;
        }
        return null;
    }

    pub fn matchTask(self: *const Registry, spec: task.TaskSpec) ?*const Entry {
        if (spec.adapter_id) |adapter_id| {
            const entry = self.findById(adapter_id) orelse return null;
            if (entry.adapter.descriptor.modality != spec.modality) return null;
            if (!entry.adapter.descriptor.supportsModelFamily(spec.model_family)) return null;
            if (!entry.adapter.descriptor.supportsOperation(spec.operation)) return null;
            return entry;
        }

        for (self.entries.items) |*entry| {
            if (entry.adapter.descriptor.modality != spec.modality) continue;
            if (!entry.adapter.descriptor.supportsModelFamily(spec.model_family)) continue;
            if (!entry.adapter.descriptor.supportsOperation(spec.operation)) continue;
            return entry;
        }

        return null;
    }
};
