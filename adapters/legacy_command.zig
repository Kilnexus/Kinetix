const std = @import("std");

pub const LegacyCommand = struct {
    allocator: std.mem.Allocator,
    workdir: []u8,
    argv: [][]u8,

    pub fn deinit(self: *LegacyCommand) void {
        self.allocator.free(self.workdir);
        for (self.argv) |arg| self.allocator.free(arg);
        self.allocator.free(self.argv);
        self.* = undefined;
    }
};

pub const BuildOptions = struct {
    operation: []const u8,
    input: ?[]const u8 = null,
    max_tokens: ?usize = null,
};

pub fn init(allocator: std.mem.Allocator, workdir: []const u8, args: []const []const u8) !LegacyCommand {
    const owned_workdir = try allocator.dupe(u8, workdir);
    errdefer allocator.free(owned_workdir);

    const argv = try allocator.alloc([]u8, args.len);
    var initialized: usize = 0;
    errdefer {
        for (argv[0..initialized]) |arg| allocator.free(arg);
        allocator.free(argv);
    }

    for (args, 0..) |arg, index| {
        argv[index] = try allocator.dupe(u8, arg);
        initialized += 1;
    }

    return .{
        .allocator = allocator,
        .workdir = owned_workdir,
        .argv = argv,
    };
}

pub fn repoRootAlloc(allocator: std.mem.Allocator) ![]u8 {
    var current = try std.fs.cwd().realpathAlloc(allocator, ".");
    errdefer allocator.free(current);

    while (true) {
        if (isRepoRoot(current)) return current;

        const parent = std.fs.path.dirname(current) orelse return error.RepoRootNotFound;
        if (std.mem.eql(u8, parent, current)) return error.RepoRootNotFound;

        const owned_parent = try allocator.dupe(u8, parent);
        allocator.free(current);
        current = owned_parent;
    }
}

pub fn legacyProjectDirAlloc(allocator: std.mem.Allocator, relative_project_dir: []const u8) ![]u8 {
    const root = try repoRootAlloc(allocator);
    defer allocator.free(root);
    return try std.fs.path.join(allocator, &.{ root, relative_project_dir });
}

fn isRepoRoot(path: []const u8) bool {
    const kinetix_path = std.fs.path.join(std.heap.page_allocator, &.{ path, "kinetix.zig" }) catch return false;
    defer std.heap.page_allocator.free(kinetix_path);
    const legacy_path = std.fs.path.join(std.heap.page_allocator, &.{ path, "legacy" }) catch return false;
    defer std.heap.page_allocator.free(legacy_path);

    return pathExists(kinetix_path, false) and pathExists(legacy_path, true);
}

fn pathExists(path: []const u8, expect_dir: bool) bool {
    if (expect_dir) {
        var dir = std.fs.openDirAbsolute(path, .{}) catch return false;
        dir.close();
        return true;
    }

    var file = std.fs.openFileAbsolute(path, .{}) catch return false;
    file.close();
    return true;
}
