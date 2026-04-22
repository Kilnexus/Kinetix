pub const kinetix = @import("kinetix_sdk");
const cli = @import("main.zig");

pub fn main(init: @import("std").process.Init) !void {
    try cli.main(init);
}
