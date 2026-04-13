const std = @import("std");
const backend = @import("../../artifacts/backend/backend.zig");

pub const ArtifactRole = backend.ArtifactRole;
pub const ArtifactLocation = backend.ArtifactLocation;
pub const WeightScheme = backend.WeightScheme;
pub const WeightSelection = backend.WeightSelection;

pub const ArtifactCatalog = struct {
    raw: backend.ModelCatalog,

    pub fn discover(backing_allocator: std.mem.Allocator, model_dir: []const u8) !ArtifactCatalog {
        return .{
            .raw = try backend.ModelCatalog.discover(backing_allocator, model_dir),
        };
    }

    pub fn deinit(self: *ArtifactCatalog) void {
        self.raw.deinit();
        self.* = undefined;
    }

    pub fn allocator(self: *const ArtifactCatalog) std.mem.Allocator {
        return self.raw.allocator;
    }

    pub fn modelDir(self: *const ArtifactCatalog) []const u8 {
        return self.raw.model_dir;
    }

    pub fn has(self: *const ArtifactCatalog, role: ArtifactRole) bool {
        return self.raw.has(role);
    }

    pub fn find(self: *const ArtifactCatalog, role: ArtifactRole) ?*const ArtifactLocation {
        return self.raw.find(role);
    }

    pub fn resolveWeights(self: *const ArtifactCatalog, preferred: WeightScheme) !WeightSelection {
        return self.raw.resolveWeights(preferred);
    }

    pub fn resolveAutoScheme(self: *const ArtifactCatalog) WeightScheme {
        return self.raw.resolveAutoScheme();
    }

    pub fn artifactCount(self: *const ArtifactCatalog) usize {
        return self.raw.artifactCount();
    }
};
