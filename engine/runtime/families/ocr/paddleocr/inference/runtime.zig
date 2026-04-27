const std = @import("std");
const imaging = @import("Pixio");
const shared_graph = @import("shared_graph");
const postprocess = @import("../postprocess/index.zig");
const preprocess = @import("../preprocess.zig");

const Tensor = shared_graph.runtime.tensor.Tensor;
const NamedTensor = shared_graph.runtime.executor.NamedTensor;
const ExecutionResult = shared_graph.runtime.executor.ExecutionResult;
const ModelMetadata = shared_graph.onnx.metadata.ModelMetadata;
const TensorInfo = shared_graph.onnx.metadata.TensorInfo;
const io = std.Options.debug_io;

pub const StageKind = enum {
    det,
    rec,
    cls,
    unknown,

    pub fn name(self: StageKind) []const u8 {
        return @tagName(self);
    }
};

pub const ImageModelResult = struct {
    allocator: std.mem.Allocator,
    input_name: []u8,
    input_shape: []usize,
    outputs: ExecutionResult,

    pub fn deinit(self: *ImageModelResult) void {
        self.allocator.free(self.input_name);
        self.allocator.free(self.input_shape);
        self.outputs.deinit();
        self.* = undefined;
    }
};

pub fn executeImageModelFile(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    image_path: []const u8,
    options: preprocess.Options,
) !ImageModelResult {
    var image = try imaging.decodeFileRgb8(allocator, image_path);
    defer image.deinit();
    return try executeImageModelFromImage(allocator, model_path, &image, options);
}

pub fn executeImageModelFromImage(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    image: *const imaging.ImageU8,
    options: preprocess.Options,
) !ImageModelResult {
    var model = try shared_graph.onnx.metadata.loadFromFile(allocator, model_path);
    defer model.deinit();
    return try executeLoadedImageModel(allocator, model_path, &model, image, options);
}

pub fn executeLoadedImageModel(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    model: *const ModelMetadata,
    image: *const imaging.ImageU8,
    options: preprocess.Options,
) !ImageModelResult {
    const input_info = firstRuntimeInput(model.graph) orelse return error.MissingOnnxGraphInput;
    const input_shape = try preprocess.shapeFromModelInput(allocator, input_info.dims);
    errdefer allocator.free(input_shape);
    const input_name = try allocator.dupe(u8, input_info.name);
    errdefer allocator.free(input_name);

    var input_tensor = try preprocess.tensorFromImage(allocator, image, input_shape, options);
    defer input_tensor.deinit();

    const external_data_dir = std.fs.path.dirname(model_path) orelse ".";
    var outputs = try shared_graph.runtime.executor.executeWithExternalData(allocator, model.graph, &.{
        NamedTensor{ .name = input_name, .tensor = input_tensor },
    }, external_data_dir);
    errdefer outputs.deinit();

    return .{
        .allocator = allocator,
        .input_name = input_name,
        .input_shape = input_shape,
        .outputs = outputs,
    };
}

pub fn findFirstOnnxModelFile(allocator: std.mem.Allocator, model_dir: []const u8) !?[]u8 {
    if (try findFirstOnnxByStage(allocator, model_dir, "det")) |path| return path;
    if (try findFirstOnnxByStage(allocator, model_dir, "rec")) |path| return path;
    if (try findFirstOnnxByStage(allocator, model_dir, "cls")) |path| return path;
    return try findFirstOnnxRecursive(allocator, model_dir, "", null, 0);
}

pub fn findFirstDictionaryFile(allocator: std.mem.Allocator, model_dir: []const u8) !?[]u8 {
    return try findFirstFileRecursive(allocator, model_dir, "", isDictionaryFile, null, 0);
}

pub fn stageFromPath(path: []const u8) StageKind {
    if (std.ascii.indexOfIgnoreCase(path, "det") != null) return .det;
    if (std.ascii.indexOfIgnoreCase(path, "rec") != null) return .rec;
    if (std.ascii.indexOfIgnoreCase(path, "cls") != null) return .cls;
    return .unknown;
}

pub fn boxesFromDetectionResult(
    allocator: std.mem.Allocator,
    result: *const ExecutionResult,
    options: postprocess.db.Options,
) ![]postprocess.db.Box {
    if (result.outputs.len == 0) return error.TensorNotFound;
    const tensor = result.outputs[0].tensor;
    const map = try probabilityMap(tensor);
    return try postprocess.db.boxesFromProbabilityMap(allocator, map.values, map.width, map.height, options);
}

pub fn decodeRecognitionResult(
    allocator: std.mem.Allocator,
    result: *const ExecutionResult,
    dictionary: []const []const u8,
) !postprocess.ctc.DecodedText {
    if (result.outputs.len == 0) return error.TensorNotFound;
    return try postprocess.ctc.decodeTensorBestPath(allocator, result.outputs[0].tensor, dictionary, .{});
}

const ProbabilityMap = struct {
    values: []const f32,
    width: usize,
    height: usize,
};

fn probabilityMap(tensor: Tensor) !ProbabilityMap {
    if (tensor.buffer != .f32) return error.UnsupportedTensorDType;
    return switch (tensor.shape.len) {
        2 => .{ .values = tensor.buffer.f32, .height = tensor.shape[0], .width = tensor.shape[1] },
        3 => blk: {
            if (tensor.shape[0] != 1) return error.UnsupportedTensorShape;
            break :blk .{ .values = tensor.buffer.f32, .height = tensor.shape[1], .width = tensor.shape[2] };
        },
        4 => blk: {
            if (tensor.shape[0] != 1 or tensor.shape[1] != 1) return error.UnsupportedTensorShape;
            break :blk .{ .values = tensor.buffer.f32, .height = tensor.shape[2], .width = tensor.shape[3] };
        },
        else => error.UnsupportedTensorRank,
    };
}

fn findFirstOnnxByStage(allocator: std.mem.Allocator, model_dir: []const u8, stage: []const u8) !?[]u8 {
    return try findFirstOnnxRecursive(allocator, model_dir, "", stage, 0);
}

fn findFirstOnnxRecursive(
    allocator: std.mem.Allocator,
    root: []const u8,
    relative: []const u8,
    stage: ?[]const u8,
    depth: usize,
) !?[]u8 {
    if (depth > 4) return null;
    const dir_path = if (relative.len == 0)
        try allocator.dupe(u8, root)
    else
        try std.fs.path.join(allocator, &.{ root, relative });
    defer allocator.free(dir_path);

    var dir = if (std.fs.path.isAbsolute(dir_path))
        std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return null
    else
        std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        switch (entry.kind) {
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".onnx")) continue;
                const candidate = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                errdefer allocator.free(candidate);
                if (stage) |stage_name| {
                    if (std.ascii.indexOfIgnoreCase(candidate, stage_name) == null) {
                        allocator.free(candidate);
                        continue;
                    }
                }
                const full_path = try std.fs.path.join(allocator, &.{ root, candidate });
                allocator.free(candidate);
                return full_path;
            },
            .directory => {
                const child = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                defer allocator.free(child);
                if (try findFirstOnnxRecursive(allocator, root, child, stage, depth + 1)) |found| return found;
            },
            else => {},
        }
    }
    return null;
}

const FilePredicate = *const fn ([]const u8) bool;

fn findFirstFileRecursive(
    allocator: std.mem.Allocator,
    root: []const u8,
    relative: []const u8,
    predicate: FilePredicate,
    stage: ?[]const u8,
    depth: usize,
) !?[]u8 {
    if (depth > 4) return null;
    const dir_path = if (relative.len == 0)
        try allocator.dupe(u8, root)
    else
        try std.fs.path.join(allocator, &.{ root, relative });
    defer allocator.free(dir_path);

    var dir = if (std.fs.path.isAbsolute(dir_path))
        std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return null
    else
        std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);

    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        switch (entry.kind) {
            .file => {
                if (!predicate(entry.name)) continue;
                const candidate = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                errdefer allocator.free(candidate);
                if (stage) |stage_name| {
                    if (std.ascii.indexOfIgnoreCase(candidate, stage_name) == null) {
                        allocator.free(candidate);
                        continue;
                    }
                }
                const full_path = try std.fs.path.join(allocator, &.{ root, candidate });
                allocator.free(candidate);
                return full_path;
            },
            .directory => {
                const child = if (relative.len == 0)
                    try allocator.dupe(u8, entry.name)
                else
                    try std.fs.path.join(allocator, &.{ relative, entry.name });
                defer allocator.free(child);
                if (try findFirstFileRecursive(allocator, root, child, predicate, stage, depth + 1)) |found| return found;
            },
            else => {},
        }
    }
    return null;
}

fn isDictionaryFile(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".txt") and std.ascii.indexOfIgnoreCase(name, "dict") != null;
}

fn firstRuntimeInput(graph: shared_graph.onnx.metadata.GraphMetadata) ?TensorInfo {
    for (graph.inputs) |input| {
        if (!isInitializer(graph, input.name)) return input;
    }
    return null;
}

fn isInitializer(graph: shared_graph.onnx.metadata.GraphMetadata, name: []const u8) bool {
    for (graph.initializers) |initializer| {
        if (std.mem.eql(u8, initializer.name, name)) return true;
    }
    return false;
}

test "paddleocr image runtime selects non-initializer graph input" {
    var model = try shared_graph.onnx.metadata.parseModel(std.testing.allocator, try identityModelBytes(std.testing.allocator));
    defer model.deinit();

    var image = try imaging.ImageU8.init(std.testing.allocator, 2, 2, 3);
    defer image.deinit();
    image.fill(255);

    var result = try executeLoadedImageModel(std.testing.allocator, "synthetic.onnx", &model, &image, .{});
    defer result.deinit();

    try std.testing.expectEqualStrings("x", result.input_name);
    try std.testing.expectEqual(@as(usize, 1), result.outputs.outputs.len);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 2, 2 }, result.outputs.outputs[0].tensor.shape);
}

fn identityModelBytes(allocator: std.mem.Allocator) ![]u8 {
    var graph = std.ArrayList(u8).init(allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "identity");
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(allocator, "x", 1, &.{ 1, 3, 2, 2 }));
    try appendOwnedMessageField(&graph, 12, try valueInfoMessage(allocator, "y", 1, &.{ 1, 3, 2, 2 }));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(allocator, "identity", "Identity", &.{"x"}, &.{"y"}));

    var model = std.ArrayList(u8).init(allocator);
    errdefer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    return try model.toOwnedSlice();
}

fn nodeMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    op_type: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (inputs) |input| try appendStringField(&out, 1, input);
    for (outputs) |output| try appendStringField(&out, 2, output);
    try appendStringField(&out, 3, name);
    try appendStringField(&out, 4, op_type);
    return try out.toOwnedSlice();
}

fn valueInfoMessage(allocator: std.mem.Allocator, name: []const u8, elem_type: u32, dims: []const i64) ![]u8 {
    var tensor_type = std.ArrayList(u8).init(allocator);
    defer tensor_type.deinit();
    try appendVarintField(&tensor_type, 1, elem_type);
    try appendOwnedMessageField(&tensor_type, 2, try shapeMessage(allocator, dims));

    var type_proto = std.ArrayList(u8).init(allocator);
    defer type_proto.deinit();
    try appendMessageField(&type_proto, 1, tensor_type.items);

    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, name);
    try appendMessageField(&out, 2, type_proto.items);
    return try out.toOwnedSlice();
}

fn shapeMessage(allocator: std.mem.Allocator, dims: []const i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (dims) |dim| {
        var dim_message = std.ArrayList(u8).init(allocator);
        defer dim_message.deinit();
        try appendVarintField(&dim_message, 1, @intCast(dim));
        try appendMessageField(&out, 1, dim_message.items);
    }
    return try out.toOwnedSlice();
}

fn appendOwnedMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []u8) !void {
    defer bytes.allocator.free(payload);
    try appendMessageField(bytes, field_number, payload);
}

fn appendMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, payload.len);
    try bytes.appendSlice(payload);
}

fn appendStringField(bytes: *std.ArrayList(u8), field_number: u64, value: []const u8) !void {
    try writeKey(bytes, field_number, 2);
    try writeVarint(bytes, value.len);
    try bytes.appendSlice(value);
}

fn appendVarintField(bytes: *std.ArrayList(u8), field_number: u64, value: u64) !void {
    try writeKey(bytes, field_number, 0);
    try writeVarint(bytes, value);
}

fn writeKey(bytes: *std.ArrayList(u8), field_number: u64, wire_type: u3) !void {
    try writeVarint(bytes, (field_number << 3) | wire_type);
}

fn writeVarint(bytes: *std.ArrayList(u8), raw: u64) !void {
    var value = raw;
    while (value >= 0x80) {
        try bytes.append(@intCast((value & 0x7f) | 0x80));
        value >>= 7;
    }
    try bytes.append(@intCast(value));
}
