const std = @import("std");
const imaging = @import("Pixio");
const preprocess = @import("../../../chandra_preprocess.zig");

pub const io = std.Options.debug_io;

pub fn isSupportedInputPath(path: []const u8) bool {
    return isRasterImagePath(path) or isFrameManifestPath(path) or isDirectoryPath(path);
}

pub fn isRasterImagePath(path: []const u8) bool {
    const extension = std.fs.path.extension(path);
    return std.ascii.eqlIgnoreCase(extension, ".png") or
        std.ascii.eqlIgnoreCase(extension, ".jpg") or
        std.ascii.eqlIgnoreCase(extension, ".jpeg") or
        std.ascii.eqlIgnoreCase(extension, ".bmp") or
        std.ascii.eqlIgnoreCase(extension, ".gif") or
        std.ascii.eqlIgnoreCase(extension, ".ico") or
        std.ascii.eqlIgnoreCase(extension, ".webp");
}

pub fn isFrameManifestPath(path: []const u8) bool {
    const extension = std.fs.path.extension(path);
    return std.ascii.eqlIgnoreCase(extension, ".frames") or
        std.ascii.eqlIgnoreCase(extension, ".txt") or
        std.ascii.eqlIgnoreCase(extension, ".lst");
}

pub fn isDirectoryPath(path: []const u8) bool {
    const dir = openDirAtPath(path, .{}) catch return false;
    var opened = dir;
    opened.close(io);
    return true;
}

pub fn loadPreparedInputFromPath(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (std.ascii.eqlIgnoreCase(std.fs.path.extension(input_path), ".gif")) {
        return try loadPreparedInputFromGif(allocator, input_path, config);
    }
    if (std.ascii.eqlIgnoreCase(std.fs.path.extension(input_path), ".webp")) {
        return try loadPreparedInputFromWebp(allocator, input_path, config);
    }
    if (isRasterImagePath(input_path)) {
        return try preprocess.loadImageInput(allocator, input_path, config);
    }
    if (isDirectoryPath(input_path)) {
        return try loadPreparedInputFromDirectory(allocator, input_path, config);
    }
    if (isFrameManifestPath(input_path)) {
        return try loadPreparedInputFromManifest(allocator, input_path, config);
    }
    return error.UnsupportedImageInput;
}

pub fn loadPreparedInputFromDirectory(
    allocator: std.mem.Allocator,
    directory_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    var dir = try openDirAtPath(directory_path, .{ .iterate = true });
    defer dir.close(io);

    var entries = std.ArrayListUnmanaged([]u8).empty;
    defer {
        for (entries.items) |entry| allocator.free(entry);
        entries.deinit(allocator);
    }

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!isRasterImagePath(entry.name)) continue;
        try entries.append(allocator, try allocator.dupe(u8, entry.name));
    }
    if (entries.items.len == 0) return error.EmptyImageSequence;

    std.sort.block([]u8, entries.items, {}, struct {
        fn lessThan(_: void, lhs: []u8, rhs: []u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    const frame_paths = try allocator.alloc([]const u8, entries.items.len);
    defer allocator.free(frame_paths);
    for (entries.items, frame_paths) |entry, *slot| {
        slot.* = try std.fs.path.join(allocator, &.{ directory_path, entry });
    }
    defer for (frame_paths) |frame_path| allocator.free(frame_path);

    return try loadPreparedInputFromResolvedPaths(allocator, frame_paths, config);
}

pub fn loadPreparedInputFromManifest(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(bytes);

    const base_dir = std.fs.path.dirname(manifest_path) orelse ".";
    var frame_paths = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (frame_paths.items) |frame_path| allocator.free(frame_path);
        frame_paths.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line_raw| {
        const line = std.mem.trim(u8, line_raw, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        const resolved = if (std.fs.path.isAbsolute(line))
            try allocator.dupe(u8, line)
        else
            try std.fs.path.join(allocator, &.{ base_dir, line });
        try frame_paths.append(allocator, resolved);
    }
    if (frame_paths.items.len == 0) return error.EmptyImageSequence;

    return try loadPreparedInputFromResolvedPaths(allocator, frame_paths.items, config);
}

pub fn openDirAtPath(path: []const u8, flags: std.Io.Dir.OpenOptions) !std.Io.Dir {
    if (std.fs.path.isAbsolute(path)) return try std.Io.Dir.openDirAbsolute(io, path, flags);
    return try std.Io.Dir.cwd().openDir(io, path, flags);
}

pub fn hasAnyFile(model_path: []const u8, names: []const []const u8) bool {
    for (names) |name| {
        if (hasFile(model_path, name)) return true;
    }
    return false;
}

pub fn hasFile(model_path: []const u8, name: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_path, .{}) catch return false;
    defer dir.close(io);

    dir.access(io, name, .{}) catch return false;
    return true;
}

fn loadPreparedInputFromGif(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) {
        var animation = try imaging.decodeFileGifFramesRgb8(allocator, input_path);
        defer animation.deinit();

        const frame_refs = try allocator.alloc(*const imaging.ImageU8, animation.frames.len);
        defer allocator.free(frame_refs);
        for (animation.frames, frame_refs) |*frame, *slot| {
            slot.* = &frame.image;
        }
        return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
    }
    return try loadPreparedInputFromSingleRaster(allocator, input_path, config);
}

fn loadPreparedInputFromWebp(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    if (comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) {
        var animation = try imaging.decodeFileWebpFramesRgb8(allocator, input_path);
        defer animation.deinit();

        const frame_refs = try allocator.alloc(*const imaging.ImageU8, animation.frames.len);
        defer allocator.free(frame_refs);
        for (animation.frames, frame_refs) |*frame, *slot| {
            slot.* = &frame.image;
        }
        return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
    }
    return try loadPreparedInputFromSingleRaster(allocator, input_path, config);
}

fn loadPreparedInputFromSingleRaster(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    return try preprocess.loadImageInput(allocator, input_path, config);
}

fn loadPreparedInputFromResolvedPaths(
    allocator: std.mem.Allocator,
    frame_paths: []const []const u8,
    config: preprocess.ImageProcessorConfig,
) !preprocess.PreparedImageInput {
    var images = std.ArrayListUnmanaged(imaging.ImageU8).empty;
    defer images.deinit(allocator);
    errdefer {
        for (images.items) |*image| image.deinit();
    }

    for (frame_paths) |frame_path| {
        try appendResolvedFramesFromPath(allocator, &images, frame_path);
    }
    if (images.items.len == 0) return error.EmptyImageSequence;

    const frame_refs = try allocator.alloc(*const imaging.ImageU8, images.items.len);
    defer allocator.free(frame_refs);
    for (images.items, frame_refs) |*image, *slot| {
        slot.* = image;
    }
    defer for (images.items) |*image| image.deinit();

    return try preprocess.prepareImageFramesInput(allocator, frame_refs, config);
}

fn appendResolvedFramesFromPath(
    allocator: std.mem.Allocator,
    images: *std.ArrayListUnmanaged(imaging.ImageU8),
    frame_path: []const u8,
) !void {
    const extension = std.fs.path.extension(frame_path);
    if (std.ascii.eqlIgnoreCase(extension, ".gif") and comptime @hasDecl(imaging, "decodeFileGifFramesRgb8")) {
        var animation = try imaging.decodeFileGifFramesRgb8(allocator, frame_path);
        defer animation.deinit();
        for (animation.frames) |*frame| {
            try images.append(allocator, try cloneImageOwned(allocator, &frame.image));
        }
        return;
    }
    if (std.ascii.eqlIgnoreCase(extension, ".webp") and comptime @hasDecl(imaging, "decodeFileWebpFramesRgb8")) {
        var animation = try imaging.decodeFileWebpFramesRgb8(allocator, frame_path);
        defer animation.deinit();
        for (animation.frames) |*frame| {
            try images.append(allocator, try cloneImageOwned(allocator, &frame.image));
        }
        return;
    }

    try images.append(allocator, try imaging.decodeFileRgb8(allocator, frame_path));
}

fn cloneImageOwned(allocator: std.mem.Allocator, source: *const imaging.ImageU8) !imaging.ImageU8 {
    var cloned = try imaging.ImageU8.init(allocator, source.width, source.height, source.channels);
    errdefer cloned.deinit();
    @memcpy(cloned.data, source.data);
    return cloned;
}
