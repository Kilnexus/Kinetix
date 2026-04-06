const cli = @import("apps/cli/main.zig");

pub fn main() !void {
    try cli.main();
}
