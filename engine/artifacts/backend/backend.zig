const std = @import("std");

pub const WeightScheme = enum {
    auto,
    bf16,
    q8,
    q6,
    q4,

    pub fn name(self: WeightScheme) []const u8 {
        return switch (self) {
            .auto => "auto",
            .bf16 => "bf16",
            .q8 => "q8",
            .q6 => "q6",
            .q4 => "q4",
        };
    }
};

pub const ArtifactRole = enum {
    config,
    tokenizer_json,
    tokenizer_model,
    vocab_json,
    vocab_txt,
    merges_txt,
    graph_json,
    weights_bin,
    ocr_model,
    safetensors,
    safetensors_index,
    q8_weights,
    q6_weights,
    q4_weights,
};

pub const ArtifactSpec = struct {
    role: ArtifactRole,
    relative_path: []const u8,
};

pub const ArtifactLocation = struct {
    role: ArtifactRole,
    relative_path: []const u8,
    absolute_path: []u8,
    owns_relative_path: bool = false,
};

pub const WeightSelection = struct {
    scheme: WeightScheme,
    path: []const u8,
};

pub const ModelCatalog = struct {
    allocator: std.mem.Allocator,
    model_dir: []u8,
    artifacts: []ArtifactLocation,

    pub fn discover(allocator: std.mem.Allocator, model_dir: []const u8) !ModelCatalog {
        const base_dir = try allocator.dupe(u8, model_dir);
        errdefer allocator.free(base_dir);

        var found = std.ArrayListUnmanaged(ArtifactLocation).empty;
        errdefer {
            for (found.items) |item| allocator.free(item.absolute_path);
            found.deinit(allocator);
        }

        inline for (known_artifacts) |spec| {
            const absolute_path = try std.fs.path.join(allocator, &.{ model_dir, spec.relative_path });
            errdefer allocator.free(absolute_path);

            if (pathExists(absolute_path)) {
                try found.append(allocator, .{
                    .role = spec.role,
                    .relative_path = spec.relative_path,
                    .absolute_path = absolute_path,
                    .owns_relative_path = false,
                });
            } else {
                allocator.free(absolute_path);
            }
        }

        try discoverExtensionArtifacts(allocator, model_dir, &found);

        return .{
            .allocator = allocator,
            .model_dir = base_dir,
            .artifacts = try found.toOwnedSlice(allocator),
        };
    }

    pub fn deinit(self: *ModelCatalog) void {
        self.allocator.free(self.model_dir);
        for (self.artifacts) |artifact| {
            if (artifact.owns_relative_path) self.allocator.free(@constCast(artifact.relative_path));
            self.allocator.free(artifact.absolute_path);
        }
        self.allocator.free(self.artifacts);
        self.* = undefined;
    }

    pub fn has(self: *const ModelCatalog, role: ArtifactRole) bool {
        return self.find(role) != null;
    }

    pub fn find(self: *const ModelCatalog, role: ArtifactRole) ?*const ArtifactLocation {
        for (self.artifacts) |*artifact| {
            if (artifact.role == role) return artifact;
        }
        return null;
    }

    pub fn resolveWeights(self: *const ModelCatalog, preferred: WeightScheme) !WeightSelection {
        const scheme = switch (preferred) {
            .auto => self.resolveAutoScheme(),
            else => preferred,
        };

        const role = roleForWeightScheme(scheme) orelse return error.InvalidWeightScheme;
        const artifact = self.find(role) orelse return error.WeightArtifactNotFound;
        return .{
            .scheme = scheme,
            .path = artifact.absolute_path,
        };
    }

    pub fn resolveAutoScheme(self: *const ModelCatalog) WeightScheme {
        if (self.has(.q8_weights)) return .q8;
        if (self.has(.q6_weights)) return .q6;
        if (self.has(.q4_weights)) return .q4;
        if (self.has(.safetensors)) return .bf16;
        return .auto;
    }

    pub fn artifactCount(self: *const ModelCatalog) usize {
        return self.artifacts.len;
    }
};

const known_artifacts = [_]ArtifactSpec{
    .{ .role = .config, .relative_path = "config.json" },
    .{ .role = .tokenizer_json, .relative_path = "tokenizer.json" },
    .{ .role = .tokenizer_model, .relative_path = "tokenizer.model" },
    .{ .role = .vocab_json, .relative_path = "vocab.json" },
    .{ .role = .vocab_txt, .relative_path = "vocab.txt" },
    .{ .role = .merges_txt, .relative_path = "merges.txt" },
    .{ .role = .graph_json, .relative_path = "graph.json" },
    .{ .role = .weights_bin, .relative_path = "weights.bin" },
    .{ .role = .safetensors, .relative_path = "model.safetensors" },
    .{ .role = .safetensors_index, .relative_path = "model.safetensors.index.json" },
    .{ .role = .q8_weights, .relative_path = "model.q8.zinfer" },
    .{ .role = .q6_weights, .relative_path = "model.q6.zinfer" },
    .{ .role = .q4_weights, .relative_path = "model.q4.zinfer" },
};

fn roleForWeightScheme(scheme: WeightScheme) ?ArtifactRole {
    return switch (scheme) {
        .bf16 => .safetensors,
        .q8 => .q8_weights,
        .q6 => .q6_weights,
        .q4 => .q4_weights,
        .auto => null,
    };
}

fn pathExists(path: []const u8) bool {
    const file = if (std.fs.path.isAbsolute(path))
        std.fs.openFileAbsolute(path, .{})
    else
        std.fs.cwd().openFile(path, .{});
    if (file) |handle| {
        handle.close();
        return true;
    } else |_| {
        return false;
    }
}

fn discoverExtensionArtifacts(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    found: *std.ArrayListUnmanaged(ArtifactLocation),
) !void {
    var dir = if (std.fs.path.isAbsolute(model_dir))
        try std.fs.openDirAbsolute(model_dir, .{ .iterate = true })
    else
        try std.fs.cwd().openDir(model_dir, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;

        if (std.mem.endsWith(u8, entry.name, ".swm") and !containsRole(found.items, .ocr_model)) {
            try appendExtensionArtifact(allocator, model_dir, found, .ocr_model, entry.name);
            continue;
        }

        if (std.mem.endsWith(u8, entry.name, ".safetensors") and !containsRelativePath(found.items, entry.name)) {
            try appendExtensionArtifact(allocator, model_dir, found, .safetensors, entry.name);
            continue;
        }
    }
}

fn appendExtensionArtifact(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    found: *std.ArrayListUnmanaged(ArtifactLocation),
    role: ArtifactRole,
    entry_name: []const u8,
) !void {
    const relative_path = try allocator.dupe(u8, entry_name);
    errdefer allocator.free(relative_path);
    const absolute_path = try std.fs.path.join(allocator, &.{ model_dir, entry_name });
    errdefer allocator.free(absolute_path);

    try found.append(allocator, .{
        .role = role,
        .relative_path = relative_path,
        .absolute_path = absolute_path,
        .owns_relative_path = true,
    });
}

fn containsRole(items: []const ArtifactLocation, role: ArtifactRole) bool {
    for (items) |item| {
        if (item.role == role) return true;
    }
    return false;
}

fn containsRelativePath(items: []const ArtifactLocation, relative_path: []const u8) bool {
    for (items) |item| {
        if (std.mem.eql(u8, item.relative_path, relative_path)) return true;
    }
    return false;
}

test "model catalog discovers sharded safetensors artifacts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeTmpFile(tmp.dir, "model-00001-of-00002.safetensors", "a");
    try writeTmpFile(tmp.dir, "model-00002-of-00002.safetensors", "b");
    try writeTmpFile(tmp.dir, "model.safetensors.index.json", "{}");

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var catalog = try ModelCatalog.discover(std.testing.allocator, root_path);
    defer catalog.deinit();

    try std.testing.expect(catalog.has(.safetensors));
    try std.testing.expect(catalog.has(.safetensors_index));
    try std.testing.expectEqual(WeightScheme.bf16, catalog.resolveAutoScheme());
}

fn writeTmpFile(dir: std.fs.Dir, relative_path: []const u8, contents: []const u8) !void {
    var file = try dir.createFile(relative_path, .{});
    defer file.close();
    try file.writeAll(contents);
}
