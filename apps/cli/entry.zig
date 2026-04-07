pub const kinetix = @import("kinetix_sdk");
const cli = @import("main.zig");

pub fn main() !void {
    try cli.main();
}
