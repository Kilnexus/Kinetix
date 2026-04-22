const std = @import("std");
const io = std.Options.debug_io;

pub const Stopwatch = struct {
    started_at: std.Io.Clock.Timestamp,

    pub fn start() Stopwatch {
        return .{
            .started_at = std.Io.Clock.Timestamp.now(io, .awake),
        };
    }

    pub fn reset(self: *Stopwatch) void {
        self.started_at = std.Io.Clock.Timestamp.now(io, .awake);
    }

    pub fn read(self: *const Stopwatch) u64 {
        return @intCast(self.started_at.untilNow(io).raw.toNanoseconds());
    }
};

pub fn start() Stopwatch {
    return Stopwatch.start();
}
