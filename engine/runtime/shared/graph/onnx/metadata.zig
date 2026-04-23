const std = @import("std");

const io = std.Options.debug_io;

pub const ElementType = struct {
    raw: u32 = 0,

    pub fn name(self: ElementType) []const u8 {
        return switch (self.raw) {
            0 => "undefined",
            1 => "float32",
            2 => "uint8",
            3 => "int8",
            4 => "uint16",
            5 => "int16",
            6 => "int32",
            7 => "int64",
            8 => "string",
            9 => "bool",
            10 => "float16",
            11 => "float64",
            12 => "uint32",
            13 => "uint64",
            16 => "bfloat16",
            else => "unknown",
        };
    }
};

pub const Dimension = union(enum) {
    value: i64,
    param: []u8,
    unknown,

    fn deinit(self: Dimension, allocator: std.mem.Allocator) void {
        switch (self) {
            .param => |text| allocator.free(text),
            else => {},
        }
    }
};

pub const TensorInfo = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    elem_type: ElementType = .{},
    dims: []Dimension = &.{},

    fn deinit(self: *TensorInfo) void {
        self.allocator.free(self.name);
        for (self.dims) |dim| dim.deinit(self.allocator);
        self.allocator.free(self.dims);
        self.* = undefined;
    }
};

pub const GraphMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    inputs: []TensorInfo = &.{},
    outputs: []TensorInfo = &.{},
    value_infos: []TensorInfo = &.{},
    initializers: []TensorInfo = &.{},
    node_count: usize = 0,

    fn deinit(self: *GraphMetadata) void {
        self.allocator.free(self.name);
        deinitTensorInfos(self.allocator, self.inputs);
        deinitTensorInfos(self.allocator, self.outputs);
        deinitTensorInfos(self.allocator, self.value_infos);
        deinitTensorInfos(self.allocator, self.initializers);
        self.* = undefined;
    }
};

pub const ModelMetadata = struct {
    allocator: std.mem.Allocator,
    ir_version: i64 = 0,
    opset_import_count: usize = 0,
    graph: GraphMetadata,

    pub fn deinit(self: *ModelMetadata) void {
        self.graph.deinit();
        self.* = undefined;
    }
};

const Reader = struct {
    bytes: []const u8,
    index: usize = 0,

    fn eof(self: Reader) bool {
        return self.index >= self.bytes.len;
    }

    fn readVarint(self: *Reader) !u64 {
        var shift: u6 = 0;
        var value: u64 = 0;
        while (self.index < self.bytes.len and shift < 64) {
            const byte = self.bytes[self.index];
            self.index += 1;
            value |= (@as(u64, byte & 0x7f) << shift);
            if ((byte & 0x80) == 0) return value;
            shift += 7;
        }
        return error.InvalidProtobufVarint;
    }

    fn readBytes(self: *Reader, len: usize) ![]const u8 {
        if (self.index + len > self.bytes.len) return error.TruncatedProtobuf;
        const out = self.bytes[self.index .. self.index + len];
        self.index += len;
        return out;
    }

    fn skip(self: *Reader, wire_type: u3) !void {
        switch (wire_type) {
            0 => _ = try self.readVarint(),
            1 => _ = try self.readBytes(8),
            2 => {
                const len: usize = @intCast(try self.readVarint());
                _ = try self.readBytes(len);
            },
            5 => _ = try self.readBytes(4),
            else => return error.UnsupportedProtobufWireType,
        }
    }
};

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !ModelMetadata {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(512 * 1024 * 1024));
    defer allocator.free(bytes);
    return try parseModel(allocator, bytes);
}

pub fn parseModel(allocator: std.mem.Allocator, bytes: []const u8) !ModelMetadata {
    var reader = Reader{ .bytes = bytes };
    var graph: ?GraphMetadata = null;
    errdefer if (graph) |*item| item.deinit();
    var ir_version: i64 = 0;
    var opset_import_count: usize = 0;

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 0) return error.InvalidOnnxModel;
                ir_version = @intCast(try reader.readVarint());
            },
            7 => {
                if (wire_type != 2) return error.InvalidOnnxModel;
                const len: usize = @intCast(try reader.readVarint());
                if (graph) |*existing| existing.deinit();
                graph = try parseGraph(allocator, try reader.readBytes(len));
            },
            8 => {
                if (wire_type != 2) return error.InvalidOnnxModel;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                opset_import_count += 1;
            },
            else => try reader.skip(wire_type),
        }
    }

    return .{
        .allocator = allocator,
        .ir_version = ir_version,
        .opset_import_count = opset_import_count,
        .graph = graph orelse return error.MissingOnnxGraph,
    };
}

fn parseGraph(allocator: std.mem.Allocator, bytes: []const u8) !GraphMetadata {
    var reader = Reader{ .bytes = bytes };
    var name = try allocator.dupe(u8, "");
    errdefer allocator.free(name);
    var inputs = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &inputs);
    var outputs = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &outputs);
    var value_infos = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &value_infos);
    var initializers = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &initializers);
    var node_count: usize = 0;

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                node_count += 1;
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(name);
                name = try allocator.dupe(u8, try reader.readBytes(len));
            },
            5 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                try initializers.append(allocator, try parseTensorProto(allocator, try reader.readBytes(len)));
            },
            11 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                try inputs.append(allocator, try parseValueInfo(allocator, try reader.readBytes(len)));
            },
            12 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                try outputs.append(allocator, try parseValueInfo(allocator, try reader.readBytes(len)));
            },
            13 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                try value_infos.append(allocator, try parseValueInfo(allocator, try reader.readBytes(len)));
            },
            else => try reader.skip(wire_type),
        }
    }

    return .{
        .allocator = allocator,
        .name = name,
        .inputs = try inputs.toOwnedSlice(allocator),
        .outputs = try outputs.toOwnedSlice(allocator),
        .value_infos = try value_infos.toOwnedSlice(allocator),
        .initializers = try initializers.toOwnedSlice(allocator),
        .node_count = node_count,
    };
}

fn parseValueInfo(allocator: std.mem.Allocator, bytes: []const u8) !TensorInfo {
    var reader = Reader{ .bytes = bytes };
    var info = TensorInfo{
        .allocator = allocator,
        .name = try allocator.dupe(u8, ""),
    };
    errdefer info.deinit();

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxValueInfo;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(info.name);
                info.name = try allocator.dupe(u8, try reader.readBytes(len));
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxValueInfo;
                const len: usize = @intCast(try reader.readVarint());
                try parseTypeProto(allocator, try reader.readBytes(len), &info);
            },
            else => try reader.skip(wire_type),
        }
    }

    return info;
}

fn parseTypeProto(allocator: std.mem.Allocator, bytes: []const u8, info: *TensorInfo) !void {
    var reader = Reader{ .bytes = bytes };
    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        if (field_number == 1 and wire_type == 2) {
            const len: usize = @intCast(try reader.readVarint());
            try parseTensorTypeProto(allocator, try reader.readBytes(len), info);
            continue;
        }
        try reader.skip(wire_type);
    }
}

fn parseTensorTypeProto(allocator: std.mem.Allocator, bytes: []const u8, info: *TensorInfo) !void {
    var reader = Reader{ .bytes = bytes };
    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 0) return error.InvalidOnnxTensorType;
                info.elem_type = .{ .raw = @intCast(try reader.readVarint()) };
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxTensorType;
                const len: usize = @intCast(try reader.readVarint());
                const dims = try parseShapeProto(allocator, try reader.readBytes(len));
                for (info.dims) |dim| dim.deinit(allocator);
                allocator.free(info.dims);
                info.dims = dims;
            },
            else => try reader.skip(wire_type),
        }
    }
}

fn parseShapeProto(allocator: std.mem.Allocator, bytes: []const u8) ![]Dimension {
    var reader = Reader{ .bytes = bytes };
    var dims = std.ArrayListUnmanaged(Dimension).empty;
    errdefer {
        for (dims.items) |dim| dim.deinit(allocator);
        dims.deinit(allocator);
    }

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        if (field_number == 1 and wire_type == 2) {
            const len: usize = @intCast(try reader.readVarint());
            try dims.append(allocator, try parseDimensionProto(allocator, try reader.readBytes(len)));
            continue;
        }
        try reader.skip(wire_type);
    }

    return try dims.toOwnedSlice(allocator);
}

fn parseDimensionProto(allocator: std.mem.Allocator, bytes: []const u8) !Dimension {
    var reader = Reader{ .bytes = bytes };
    var out: Dimension = .unknown;
    errdefer out.deinit(allocator);

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 0) return error.InvalidOnnxDimension;
                out.deinit(allocator);
                out = .{ .value = @intCast(try reader.readVarint()) };
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxDimension;
                const len: usize = @intCast(try reader.readVarint());
                out.deinit(allocator);
                out = .{ .param = try allocator.dupe(u8, try reader.readBytes(len)) };
            },
            else => try reader.skip(wire_type),
        }
    }

    return out;
}

fn parseTensorProto(allocator: std.mem.Allocator, bytes: []const u8) !TensorInfo {
    var reader = Reader{ .bytes = bytes };
    var info = TensorInfo{
        .allocator = allocator,
        .name = try allocator.dupe(u8, ""),
    };
    errdefer info.deinit();
    var dims = std.ArrayListUnmanaged(Dimension).empty;
    errdefer {
        for (dims.items) |dim| dim.deinit(allocator);
        dims.deinit(allocator);
    }

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 0) return error.InvalidOnnxTensor;
                try dims.append(allocator, .{ .value = @intCast(try reader.readVarint()) });
            },
            2 => {
                if (wire_type != 0) return error.InvalidOnnxTensor;
                info.elem_type = .{ .raw = @intCast(try reader.readVarint()) };
            },
            8 => {
                if (wire_type != 2) return error.InvalidOnnxTensor;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(info.name);
                info.name = try allocator.dupe(u8, try reader.readBytes(len));
            },
            else => try reader.skip(wire_type),
        }
    }

    info.dims = try dims.toOwnedSlice(allocator);
    return info;
}

fn deinitTensorInfos(allocator: std.mem.Allocator, items: []TensorInfo) void {
    for (items) |*item| item.deinit();
    allocator.free(items);
}

fn deinitTensorInfoList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(TensorInfo)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

test "onnx metadata parser extracts graph inputs outputs and initializers" {
    var graph = std.ArrayList(u8).init(std.testing.allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "moss_prefill");
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(std.testing.allocator, "input_ids", 6, &.{ 1, 8, 17 }));
    try appendOwnedMessageField(&graph, 12, try valueInfoMessage(std.testing.allocator, "global_hidden", 1, &.{ 1, 768 }));
    try appendOwnedMessageField(&graph, 13, try valueInfoMessage(std.testing.allocator, "symbolic", 1, &.{ -1, 768 }));
    try appendOwnedMessageField(&graph, 5, try tensorMessage(std.testing.allocator, "embed.weight", 1, &.{ 16384, 768 }));
    try appendMessageField(&graph, 1, &.{}); // one node body, intentionally skipped.

    var model = std.ArrayList(u8).init(std.testing.allocator);
    defer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    try appendMessageField(&model, 8, &.{});

    var parsed = try parseModel(std.testing.allocator, model.items);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(i64, 8), parsed.ir_version);
    try std.testing.expectEqual(@as(usize, 1), parsed.opset_import_count);
    try std.testing.expectEqualStrings("moss_prefill", parsed.graph.name);
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.node_count);
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.inputs.len);
    try std.testing.expectEqualStrings("input_ids", parsed.graph.inputs[0].name);
    try std.testing.expectEqual(@as(u32, 6), parsed.graph.inputs[0].elem_type.raw);
    try std.testing.expectEqual(@as(i64, 17), parsed.graph.inputs[0].dims[2].value);
    try std.testing.expectEqualStrings("global_hidden", parsed.graph.outputs[0].name);
    try std.testing.expectEqualStrings("embed.weight", parsed.graph.initializers[0].name);
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
        if (dim < 0) {
            try appendStringField(&dim_message, 2, "batch");
        } else {
            try appendVarintField(&dim_message, 1, @intCast(dim));
        }
        try appendMessageField(&out, 1, dim_message.items);
    }
    return try out.toOwnedSlice();
}

fn tensorMessage(allocator: std.mem.Allocator, name: []const u8, elem_type: u32, dims: []const i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (dims) |dim| try appendVarintField(&out, 1, @intCast(dim));
    try appendVarintField(&out, 2, elem_type);
    try appendStringField(&out, 8, name);
    return try out.toOwnedSlice();
}

fn appendOwnedMessageField(bytes: *std.ArrayList(u8), field_number: u64, payload: []u8) !void {
    defer std.testing.allocator.free(payload);
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
