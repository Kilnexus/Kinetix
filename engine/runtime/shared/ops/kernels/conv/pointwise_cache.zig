const std = @import("std");
const common = @import("common.zig");
const io = std.Options.debug_io;

pub const PointwisePackKey = struct {
    ptr: usize,
    out_channels: usize,
    in_per_group: usize,
    groups: usize,
};

pub const PackedPointwiseWeights = struct {
    out_channels: usize,
    in_per_group: usize,
    groups: usize,
    out_per_group: usize,
    data: []f32,
};

pub const PointwisePackCache = struct {
    const threadlocal_capacity = 16;

    var mutex: std.Io.Mutex = .init;
    var cache: std.AutoHashMapUnmanaged(PointwisePackKey, PackedPointwiseWeights) = .{};
    threadlocal var tl_len: usize = 0;
    threadlocal var tl_next: usize = 0;
    threadlocal var tl_keys: [threadlocal_capacity]PointwisePackKey = undefined;
    threadlocal var tl_values: [threadlocal_capacity]PackedPointwiseWeights = undefined;

    pub fn get(weights: *const common.Tensor, groups: usize) ?PackedPointwiseWeights {
        const key = makeKey(weights, groups) orelse return null;
        if (getThreadLocal(key)) |cached| return cached;

        mutex.lockUncancelable(io);
        defer mutex.unlock(io);

        if (cache.get(key)) |cached| {
            putThreadLocal(key, cached);
            return cached;
        }

        const pack_weights = build(weights, groups) orelse return null;
        cache.put(std.heap.page_allocator, key, pack_weights) catch {
            std.heap.page_allocator.free(pack_weights.data);
            return null;
        };
        const cached = cache.get(key).?;
        putThreadLocal(key, cached);
        return cached;
    }

    fn makeKey(weights: *const common.Tensor, groups: usize) ?PointwisePackKey {
        if (groups == 0) return null;
        const out_channels = weights.shape[0];
        const in_per_group = weights.shape[1];
        if (out_channels == 0 or in_per_group == 0) return null;
        if (out_channels % groups != 0) return null;

        return .{
            .ptr = @intFromPtr(weights.data.ptr),
            .out_channels = out_channels,
            .in_per_group = in_per_group,
            .groups = groups,
        };
    }

    fn keyEqual(lhs: PointwisePackKey, rhs: PointwisePackKey) bool {
        return lhs.ptr == rhs.ptr and
            lhs.out_channels == rhs.out_channels and
            lhs.in_per_group == rhs.in_per_group and
            lhs.groups == rhs.groups;
    }

    fn getThreadLocal(key: PointwisePackKey) ?PackedPointwiseWeights {
        for (0..tl_len) |index| {
            if (keyEqual(tl_keys[index], key)) {
                return tl_values[index];
            }
        }
        return null;
    }

    fn putThreadLocal(key: PointwisePackKey, pack_weights: PackedPointwiseWeights) void {
        for (0..tl_len) |index| {
            if (keyEqual(tl_keys[index], key)) {
                tl_values[index] = pack_weights;
                return;
            }
        }

        if (tl_len < threadlocal_capacity) {
            const index = tl_len;
            tl_keys[index] = key;
            tl_values[index] = pack_weights;
            tl_len += 1;
            return;
        }

        const replace_index = tl_next;
        tl_keys[replace_index] = key;
        tl_values[replace_index] = pack_weights;
        tl_next = (tl_next + 1) % threadlocal_capacity;
    }

    fn build(weights: *const common.Tensor, groups: usize) ?PackedPointwiseWeights {
        const out_channels = weights.shape[0];
        const in_per_group = weights.shape[1];
        if (groups == 0 or out_channels % groups != 0) return null;
        const out_per_group = out_channels / groups;
        const total_weights = out_channels * in_per_group;

        const packed_data = std.heap.page_allocator.alloc(f32, total_weights) catch return null;
        errdefer std.heap.page_allocator.free(packed_data);

        for (0..groups) |group_idx| {
            for (0..in_per_group) |ic_local| {
                const packed_row_base = (group_idx * in_per_group + ic_local) * out_per_group;
                for (0..out_per_group) |oc_local| {
                    const oc = group_idx * out_per_group + oc_local;
                    packed_data[packed_row_base + oc_local] = weights.data[oc * in_per_group + ic_local];
                }
            }
        }

        return .{
            .out_channels = out_channels,
            .in_per_group = in_per_group,
            .groups = groups,
            .out_per_group = out_per_group,
            .data = packed_data,
        };
    }
};
