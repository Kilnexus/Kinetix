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
    raw_data_len: usize = 0,
    data_location: u32 = 0,
    external_data: []ExternalDataEntry = &.{},

    fn deinit(self: *TensorInfo) void {
        self.allocator.free(self.name);
        for (self.dims) |dim| dim.deinit(self.allocator);
        self.allocator.free(self.dims);
        for (self.external_data) |*entry| entry.deinit();
        self.allocator.free(self.external_data);
        self.* = undefined;
    }

    pub fn isExternal(self: TensorInfo) bool {
        return self.data_location == 1 or self.external_data.len != 0;
    }

    pub fn externalValue(self: TensorInfo, key: []const u8) ?[]const u8 {
        for (self.external_data) |entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry.value;
        }
        return null;
    }
};

pub const ExternalDataEntry = struct {
    allocator: std.mem.Allocator,
    key: []u8,
    value: []u8,

    fn deinit(self: *ExternalDataEntry) void {
        self.allocator.free(self.key);
        self.allocator.free(self.value);
        self.* = undefined;
    }
};

pub const OpsetImport = struct {
    allocator: std.mem.Allocator,
    domain: []u8,
    version: i64 = 0,

    fn deinit(self: *OpsetImport) void {
        self.allocator.free(self.domain);
        self.* = undefined;
    }
};

pub const AttributeInfo = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    kind: u32 = 0,
    float_count: usize = 0,
    int_count: usize = 0,
    string_count: usize = 0,
    tensor_count: usize = 0,
    graph_count: usize = 0,

    fn deinit(self: *AttributeInfo) void {
        self.allocator.free(self.name);
        self.* = undefined;
    }
};

pub const NodeInfo = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    op_type: []u8,
    domain: []u8,
    inputs: []const []const u8 = &.{},
    outputs: []const []const u8 = &.{},
    attributes: []AttributeInfo = &.{},

    fn deinit(self: *NodeInfo) void {
        self.allocator.free(self.name);
        self.allocator.free(self.op_type);
        self.allocator.free(self.domain);
        freeStringList(self.allocator, self.inputs);
        freeStringList(self.allocator, self.outputs);
        for (self.attributes) |*attribute| attribute.deinit();
        self.allocator.free(self.attributes);
        self.* = undefined;
    }

    pub fn inputCount(self: NodeInfo) usize {
        return self.inputs.len;
    }

    pub fn outputCount(self: NodeInfo) usize {
        return self.outputs.len;
    }
};

pub const GraphMetadata = struct {
    allocator: std.mem.Allocator,
    name: []u8,
    nodes: []NodeInfo = &.{},
    inputs: []TensorInfo = &.{},
    outputs: []TensorInfo = &.{},
    value_infos: []TensorInfo = &.{},
    initializers: []TensorInfo = &.{},
    node_count: usize = 0,

    fn deinit(self: *GraphMetadata) void {
        self.allocator.free(self.name);
        for (self.nodes) |*node| node.deinit();
        self.allocator.free(self.nodes);
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
    opsets: []OpsetImport = &.{},
    opset_import_count: usize = 0,
    graph: GraphMetadata,

    pub fn deinit(self: *ModelMetadata) void {
        for (self.opsets) |*opset| opset.deinit();
        self.allocator.free(self.opsets);
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
    var opsets = std.ArrayListUnmanaged(OpsetImport).empty;
    errdefer deinitOpsetList(allocator, &opsets);

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
                try opsets.append(allocator, try parseOpsetImport(allocator, try reader.readBytes(len)));
            },
            else => try reader.skip(wire_type),
        }
    }

    const parsed_graph = graph orelse return error.MissingOnnxGraph;
    const owned_opsets = try opsets.toOwnedSlice(allocator);
    return .{
        .allocator = allocator,
        .ir_version = ir_version,
        .opsets = owned_opsets,
        .opset_import_count = owned_opsets.len,
        .graph = parsed_graph,
    };
}

fn parseOpsetImport(allocator: std.mem.Allocator, bytes: []const u8) !OpsetImport {
    var reader = Reader{ .bytes = bytes };
    var out = OpsetImport{
        .allocator = allocator,
        .domain = try allocator.dupe(u8, ""),
    };
    errdefer out.deinit();

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxOpsetImport;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(out.domain);
                out.domain = try allocator.dupe(u8, try reader.readBytes(len));
            },
            2 => {
                if (wire_type != 0) return error.InvalidOnnxOpsetImport;
                out.version = @intCast(try reader.readVarint());
            },
            else => try reader.skip(wire_type),
        }
    }

    return out;
}

fn parseGraph(allocator: std.mem.Allocator, bytes: []const u8) !GraphMetadata {
    var reader = Reader{ .bytes = bytes };
    var name = try allocator.dupe(u8, "");
    errdefer allocator.free(name);
    var nodes = std.ArrayListUnmanaged(NodeInfo).empty;
    errdefer deinitNodeList(allocator, &nodes);
    var inputs = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &inputs);
    var outputs = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &outputs);
    var value_infos = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &value_infos);
    var initializers = std.ArrayListUnmanaged(TensorInfo).empty;
    errdefer deinitTensorInfoList(allocator, &initializers);

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxGraph;
                const len: usize = @intCast(try reader.readVarint());
                try nodes.append(allocator, try parseNode(allocator, try reader.readBytes(len)));
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

    const owned_nodes = try nodes.toOwnedSlice(allocator);
    return .{
        .allocator = allocator,
        .name = name,
        .nodes = owned_nodes,
        .inputs = try inputs.toOwnedSlice(allocator),
        .outputs = try outputs.toOwnedSlice(allocator),
        .value_infos = try value_infos.toOwnedSlice(allocator),
        .initializers = try initializers.toOwnedSlice(allocator),
        .node_count = owned_nodes.len,
    };
}

fn parseNode(allocator: std.mem.Allocator, bytes: []const u8) !NodeInfo {
    var reader = Reader{ .bytes = bytes };
    var node = NodeInfo{
        .allocator = allocator,
        .name = try allocator.dupe(u8, ""),
        .op_type = try allocator.dupe(u8, ""),
        .domain = try allocator.dupe(u8, ""),
    };
    errdefer node.deinit();
    var inputs = std.ArrayListUnmanaged([]const u8).empty;
    errdefer deinitStringList(allocator, &inputs);
    var outputs = std.ArrayListUnmanaged([]const u8).empty;
    errdefer deinitStringList(allocator, &outputs);
    var attributes = std.ArrayListUnmanaged(AttributeInfo).empty;
    errdefer deinitAttributeList(allocator, &attributes);

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                try inputs.append(allocator, try allocator.dupe(u8, try reader.readBytes(len)));
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                try outputs.append(allocator, try allocator.dupe(u8, try reader.readBytes(len)));
            },
            3 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(node.name);
                node.name = try allocator.dupe(u8, try reader.readBytes(len));
            },
            4 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(node.op_type);
                node.op_type = try allocator.dupe(u8, try reader.readBytes(len));
            },
            5 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                try attributes.append(allocator, try parseAttribute(allocator, try reader.readBytes(len)));
            },
            7 => {
                if (wire_type != 2) return error.InvalidOnnxNode;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(node.domain);
                node.domain = try allocator.dupe(u8, try reader.readBytes(len));
            },
            else => try reader.skip(wire_type),
        }
    }

    node.inputs = try inputs.toOwnedSlice(allocator);
    node.outputs = try outputs.toOwnedSlice(allocator);
    node.attributes = try attributes.toOwnedSlice(allocator);
    return node;
}

fn parseAttribute(allocator: std.mem.Allocator, bytes: []const u8) !AttributeInfo {
    var reader = Reader{ .bytes = bytes };
    var attribute = AttributeInfo{
        .allocator = allocator,
        .name = try allocator.dupe(u8, ""),
    };
    errdefer attribute.deinit();

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxAttribute;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(attribute.name);
                attribute.name = try allocator.dupe(u8, try reader.readBytes(len));
            },
            2 => {
                if (wire_type != 5) return error.InvalidOnnxAttribute;
                _ = try reader.readBytes(4);
                attribute.float_count = 1;
            },
            3 => {
                if (wire_type != 0) return error.InvalidOnnxAttribute;
                _ = try reader.readVarint();
                attribute.int_count = 1;
            },
            4 => {
                if (wire_type != 2) return error.InvalidOnnxAttribute;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                attribute.string_count = 1;
            },
            5, 10 => {
                if (wire_type != 2) return error.InvalidOnnxAttribute;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                attribute.tensor_count += 1;
            },
            6, 11 => {
                if (wire_type != 2) return error.InvalidOnnxAttribute;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                attribute.graph_count += 1;
            },
            7 => {
                if (wire_type == 2) {
                    const len: usize = @intCast(try reader.readVarint());
                    _ = try reader.readBytes(len);
                    attribute.float_count += len / 4;
                } else if (wire_type == 5) {
                    _ = try reader.readBytes(4);
                    attribute.float_count += 1;
                } else return error.InvalidOnnxAttribute;
            },
            8 => {
                if (wire_type == 2) {
                    var packed_reader = Reader{ .bytes = try reader.readBytes(@intCast(try reader.readVarint())) };
                    while (!packed_reader.eof()) {
                        _ = try packed_reader.readVarint();
                        attribute.int_count += 1;
                    }
                } else if (wire_type == 0) {
                    _ = try reader.readVarint();
                    attribute.int_count += 1;
                } else return error.InvalidOnnxAttribute;
            },
            9 => {
                if (wire_type != 2) return error.InvalidOnnxAttribute;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                attribute.string_count += 1;
            },
            20 => {
                if (wire_type != 0) return error.InvalidOnnxAttribute;
                attribute.kind = @intCast(try reader.readVarint());
            },
            else => try reader.skip(wire_type),
        }
    }

    return attribute;
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
    var external_data = std.ArrayListUnmanaged(ExternalDataEntry).empty;
    errdefer deinitExternalDataList(allocator, &external_data);

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
            9 => {
                if (wire_type != 2) return error.InvalidOnnxTensor;
                const len: usize = @intCast(try reader.readVarint());
                _ = try reader.readBytes(len);
                info.raw_data_len = len;
            },
            13 => {
                if (wire_type != 2) return error.InvalidOnnxTensor;
                const len: usize = @intCast(try reader.readVarint());
                try external_data.append(allocator, try parseExternalDataEntry(allocator, try reader.readBytes(len)));
            },
            14 => {
                if (wire_type != 0) return error.InvalidOnnxTensor;
                info.data_location = @intCast(try reader.readVarint());
            },
            else => try reader.skip(wire_type),
        }
    }

    info.dims = try dims.toOwnedSlice(allocator);
    info.external_data = try external_data.toOwnedSlice(allocator);
    return info;
}

fn parseExternalDataEntry(allocator: std.mem.Allocator, bytes: []const u8) !ExternalDataEntry {
    var reader = Reader{ .bytes = bytes };
    var entry = ExternalDataEntry{
        .allocator = allocator,
        .key = try allocator.dupe(u8, ""),
        .value = try allocator.dupe(u8, ""),
    };
    errdefer entry.deinit();

    while (!reader.eof()) {
        const key = try reader.readVarint();
        const field_number = key >> 3;
        const wire_type: u3 = @intCast(key & 0x07);

        switch (field_number) {
            1 => {
                if (wire_type != 2) return error.InvalidOnnxExternalData;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(entry.key);
                entry.key = try allocator.dupe(u8, try reader.readBytes(len));
            },
            2 => {
                if (wire_type != 2) return error.InvalidOnnxExternalData;
                const len: usize = @intCast(try reader.readVarint());
                allocator.free(entry.value);
                entry.value = try allocator.dupe(u8, try reader.readBytes(len));
            },
            else => try reader.skip(wire_type),
        }
    }

    return entry;
}

fn deinitTensorInfos(allocator: std.mem.Allocator, items: []TensorInfo) void {
    for (items) |*item| item.deinit();
    allocator.free(items);
}

fn deinitTensorInfoList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(TensorInfo)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

fn deinitOpsetList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(OpsetImport)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

fn deinitNodeList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(NodeInfo)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

fn deinitAttributeList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(AttributeInfo)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

fn deinitExternalDataList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged(ExternalDataEntry)) void {
    for (list.items) |*item| item.deinit();
    list.deinit(allocator);
}

fn deinitStringList(allocator: std.mem.Allocator, list: *std.ArrayListUnmanaged([]const u8)) void {
    for (list.items) |item| allocator.free(item);
    list.deinit(allocator);
}

fn freeStringList(allocator: std.mem.Allocator, items: []const []const u8) void {
    for (items) |item| allocator.free(item);
    allocator.free(items);
}

test "onnx metadata parser extracts graph inputs outputs and initializers" {
    var graph = std.ArrayList(u8).init(std.testing.allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "moss_prefill");
    try appendOwnedMessageField(&graph, 11, try valueInfoMessage(std.testing.allocator, "input_ids", 6, &.{ 1, 8, 17 }));
    try appendOwnedMessageField(&graph, 12, try valueInfoMessage(std.testing.allocator, "global_hidden", 1, &.{ 1, 768 }));
    try appendOwnedMessageField(&graph, 13, try valueInfoMessage(std.testing.allocator, "symbolic", 1, &.{ -1, 768 }));
    try appendOwnedMessageField(&graph, 5, try tensorMessage(std.testing.allocator, "embed.weight", 1, &.{ 16384, 768 }));
    try appendOwnedMessageField(&graph, 1, try nodeMessage(
        std.testing.allocator,
        "matmul_0",
        "MatMul",
        &.{ "input_ids", "embed.weight" },
        &.{"hidden"},
        &.{},
    ));

    var model = std.ArrayList(u8).init(std.testing.allocator);
    defer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    try appendOwnedMessageField(&model, 8, try opsetImportMessage(std.testing.allocator, "", 17));

    var parsed = try parseModel(std.testing.allocator, model.items);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(i64, 8), parsed.ir_version);
    try std.testing.expectEqual(@as(usize, 1), parsed.opset_import_count);
    try std.testing.expectEqual(@as(i64, 17), parsed.opsets[0].version);
    try std.testing.expectEqualStrings("moss_prefill", parsed.graph.name);
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.node_count);
    try std.testing.expectEqualStrings("MatMul", parsed.graph.nodes[0].op_type);
    try std.testing.expectEqualStrings("input_ids", parsed.graph.nodes[0].inputs[0]);
    try std.testing.expectEqualStrings("hidden", parsed.graph.nodes[0].outputs[0]);
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.inputs.len);
    try std.testing.expectEqualStrings("input_ids", parsed.graph.inputs[0].name);
    try std.testing.expectEqual(@as(u32, 6), parsed.graph.inputs[0].elem_type.raw);
    try std.testing.expectEqual(@as(i64, 17), parsed.graph.inputs[0].dims[2].value);
    try std.testing.expectEqualStrings("global_hidden", parsed.graph.outputs[0].name);
    try std.testing.expectEqualStrings("embed.weight", parsed.graph.initializers[0].name);
}

test "onnx metadata parser extracts node attributes and domains" {
    var graph = std.ArrayList(u8).init(std.testing.allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "ops");
    const allowzero_attr = try attributeIntMessage(std.testing.allocator, "allowzero", 1);
    defer std.testing.allocator.free(allowzero_attr);
    try appendOwnedMessageField(&graph, 1, try nodeMessage(
        std.testing.allocator,
        "reshape_0",
        "Reshape",
        &.{ "input", "shape" },
        &.{"reshaped"},
        &.{allowzero_attr},
    ));

    var model = std.ArrayList(u8).init(std.testing.allocator);
    defer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);
    try appendOwnedMessageField(&model, 8, try opsetImportMessage(std.testing.allocator, "ai.onnx", 19));

    var parsed = try parseModel(std.testing.allocator, model.items);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("ai.onnx", parsed.opsets[0].domain);
    try std.testing.expectEqualStrings("reshape_0", parsed.graph.nodes[0].name);
    try std.testing.expectEqualStrings("Reshape", parsed.graph.nodes[0].op_type);
    try std.testing.expectEqual(@as(usize, 2), parsed.graph.nodes[0].inputCount());
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.nodes[0].outputCount());
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.nodes[0].attributes.len);
    try std.testing.expectEqualStrings("allowzero", parsed.graph.nodes[0].attributes[0].name);
    try std.testing.expectEqual(@as(usize, 1), parsed.graph.nodes[0].attributes[0].int_count);
}

test "onnx metadata parser extracts initializer external data contract" {
    var graph = std.ArrayList(u8).init(std.testing.allocator);
    defer graph.deinit();
    try appendStringField(&graph, 2, "external");
    try appendOwnedMessageField(&graph, 5, try tensorExternalMessage(
        std.testing.allocator,
        "linear.weight",
        1,
        &.{ 768, 768 },
        "weights.data",
        128,
        2359296,
    ));

    var model = std.ArrayList(u8).init(std.testing.allocator);
    defer model.deinit();
    try appendVarintField(&model, 1, 8);
    try appendMessageField(&model, 7, graph.items);

    var parsed = try parseModel(std.testing.allocator, model.items);
    defer parsed.deinit();

    const tensor = parsed.graph.initializers[0];
    try std.testing.expect(tensor.isExternal());
    try std.testing.expectEqual(@as(u32, 1), tensor.data_location);
    try std.testing.expectEqualStrings("weights.data", tensor.externalValue("location").?);
    try std.testing.expectEqualStrings("128", tensor.externalValue("offset").?);
    try std.testing.expectEqualStrings("2359296", tensor.externalValue("length").?);
}

fn opsetImportMessage(allocator: std.mem.Allocator, domain: []const u8, version: i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, domain);
    try appendVarintField(&out, 2, @intCast(version));
    return try out.toOwnedSlice();
}

fn nodeMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    op_type: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
    attributes: []const []const u8,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (inputs) |input| try appendStringField(&out, 1, input);
    for (outputs) |output| try appendStringField(&out, 2, output);
    try appendStringField(&out, 3, name);
    try appendStringField(&out, 4, op_type);
    for (attributes) |attribute| try appendMessageField(&out, 5, attribute);
    return try out.toOwnedSlice();
}

fn attributeIntMessage(allocator: std.mem.Allocator, name: []const u8, value: i64) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, name);
    try appendVarintField(&out, 3, @intCast(value));
    try appendVarintField(&out, 20, 2);
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

fn tensorExternalMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    elem_type: u32,
    dims: []const i64,
    location: []const u8,
    offset: u64,
    length: u64,
) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    for (dims) |dim| try appendVarintField(&out, 1, @intCast(dim));
    try appendVarintField(&out, 2, elem_type);
    try appendStringField(&out, 8, name);
    try appendOwnedMessageField(&out, 13, try stringStringEntryMessage(allocator, "location", location));
    var offset_buffer: [32]u8 = undefined;
    try appendOwnedMessageField(&out, 13, try stringStringEntryMessage(
        allocator,
        "offset",
        try std.fmt.bufPrint(&offset_buffer, "{d}", .{offset}),
    ));
    var length_buffer: [32]u8 = undefined;
    try appendOwnedMessageField(&out, 13, try stringStringEntryMessage(
        allocator,
        "length",
        try std.fmt.bufPrint(&length_buffer, "{d}", .{length}),
    ));
    try appendVarintField(&out, 14, 1);
    return try out.toOwnedSlice();
}

fn stringStringEntryMessage(allocator: std.mem.Allocator, key: []const u8, value: []const u8) ![]u8 {
    var out = std.ArrayList(u8).init(allocator);
    errdefer out.deinit();
    try appendStringField(&out, 1, key);
    try appendStringField(&out, 2, value);
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
