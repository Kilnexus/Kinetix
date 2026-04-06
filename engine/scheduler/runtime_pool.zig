const std = @import("std");

pub fn RuntimePool(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Lease = struct {
            pool: *Self,
            index: usize,
            item: *T,

            pub fn release(self: Lease) void {
                self.pool.release(self.index);
            }
        };

        allocator: std.mem.Allocator,
        items: []T,
        available: []bool,
        mutex: std.Thread.Mutex = .{},
        condition: std.Thread.Condition = .{},

        pub fn initOwned(allocator: std.mem.Allocator, items: []T) !Self {
            const available = try allocator.alloc(bool, items.len);
            @memset(available, true);
            return .{
                .allocator = allocator,
                .items = items,
                .available = available,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.available);
            self.allocator.free(self.items);
            self.* = undefined;
        }

        pub fn len(self: *const Self) usize {
            return self.items.len;
        }

        pub fn availableCount(self: *Self) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.availableCountLocked();
        }

        pub fn tryAcquire(self: *Self) ?Lease {
            self.mutex.lock();
            defer self.mutex.unlock();

            for (self.available, 0..) |is_available, idx| {
                if (!is_available) continue;
                self.available[idx] = false;
                return .{
                    .pool = self,
                    .index = idx,
                    .item = &self.items[idx],
                };
            }

            return null;
        }

        pub fn acquire(self: *Self) Lease {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (true) {
                for (self.available, 0..) |is_available, idx| {
                    if (!is_available) continue;
                    self.available[idx] = false;
                    return .{
                        .pool = self,
                        .index = idx,
                        .item = &self.items[idx],
                    };
                }
                self.condition.wait(&self.mutex);
            }
        }

        pub fn release(self: *Self, index: usize) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            std.debug.assert(index < self.available.len);
            std.debug.assert(!self.available[index]);
            self.available[index] = true;
            self.condition.signal();
        }

        fn availableCountLocked(self: *const Self) usize {
            var count: usize = 0;
            for (self.available) |is_available| {
                count += @intFromBool(is_available);
            }
            return count;
        }
    };
}
