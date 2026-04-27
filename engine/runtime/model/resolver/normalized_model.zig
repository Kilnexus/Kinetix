const std = @import("std");
const catalog_mod = @import("../../catalog/catalog.zig");
const locator = @import("../../catalog/locator.zig");
const report_mod = @import("support_report.zig");
const types = @import("../../types.zig");

pub const RuntimeModelDescriptor = struct {
    allocator: std.mem.Allocator,
    id: []u8,
    modality: types.Modality,
    family: []u8,
    variant: ?[]u8 = null,
    source_format: types.SourceFormat,
    normalized_format: types.NormalizedFormat,

    pub fn deinit(self: *RuntimeModelDescriptor) void {
        self.allocator.free(self.id);
        self.allocator.free(self.family);
        if (self.variant) |value| self.allocator.free(value);
        self.* = undefined;
    }
};

pub const RuntimeArtifactSet = struct {
    allocator: std.mem.Allocator,
    model_dir: []u8,
    config_path: ?[]u8 = null,
    tokenizer_path: ?[]u8 = null,
    graph_path: ?[]u8 = null,
    weights_path: ?[]u8 = null,
    binary_weights_path: ?[]u8 = null,
    ocr_model_path: ?[]u8 = null,

    pub const InitOptions = struct {
        selected_weights_path: ?[]const u8 = null,
    };

    pub fn initFromCatalog(
        allocator: std.mem.Allocator,
        catalog: *const catalog_mod.ArtifactCatalog,
        options: InitOptions,
    ) !RuntimeArtifactSet {
        return .{
            .allocator = allocator,
            .model_dir = try allocator.dupe(u8, catalog.modelDir()),
            .config_path = try locator.duplicatePath(allocator, if (catalog.find(.config)) |item| item.absolute_path else null),
            .tokenizer_path = try pickTokenizerPath(allocator, catalog),
            .graph_path = try locator.duplicatePath(allocator, if (catalog.find(.graph_json)) |item| item.absolute_path else null),
            .weights_path = try locator.duplicatePath(allocator, options.selected_weights_path),
            .binary_weights_path = try locator.duplicatePath(allocator, if (catalog.find(.weights_bin)) |item| item.absolute_path else null),
            .ocr_model_path = try locator.duplicatePath(allocator, if (catalog.find(.ocr_model)) |item| item.absolute_path else null),
        };
    }

    pub fn deinit(self: *RuntimeArtifactSet) void {
        self.allocator.free(self.model_dir);
        freeOptional(self.allocator, self.config_path);
        freeOptional(self.allocator, self.tokenizer_path);
        freeOptional(self.allocator, self.graph_path);
        freeOptional(self.allocator, self.weights_path);
        freeOptional(self.allocator, self.binary_weights_path);
        freeOptional(self.allocator, self.ocr_model_path);
        self.* = undefined;
    }
};

pub const RuntimeCapabilitySet = struct {
    supports_sync: bool = true,
    supports_async: bool = false,
    supports_stream: bool = false,
    supports_batch: bool = false,
    supports_native_exec: bool = false,
    supported_operations: []const []const u8 = &.{},
    supported_operation_ids: []const types.RuntimeOperation = &.{},
    accepted_inputs: []const types.InputKind = &.{},

    pub fn supportsOperation(self: RuntimeCapabilitySet, operation: []const u8, operation_id: types.RuntimeOperation) bool {
        if (self.supported_operation_ids.len != 0) {
            for (self.supported_operation_ids) |supported| {
                if (supported == operation_id) return true;
            }
            return false;
        }

        if (self.supported_operations.len == 0) return true;
        for (self.supported_operations) |supported| {
            if (std.mem.eql(u8, supported, operation)) return true;
        }
        return false;
    }

    pub fn acceptsInput(self: RuntimeCapabilitySet, kind: types.InputKind) bool {
        if (kind == .none) return true;
        if (self.accepted_inputs.len == 0) return true;
        for (self.accepted_inputs) |accepted| {
            if (accepted == kind) return true;
        }
        return false;
    }

    pub fn abiCapabilities(self: RuntimeCapabilitySet) types.RuntimeAbi.CapabilitySet {
        return .{
            .flags = .{
                .sync = self.supports_sync,
                .async_exec = self.supports_async,
                .stream = self.supports_stream,
                .batch = self.supports_batch,
                .native_exec = self.supports_native_exec,
            },
            .operations = self.supported_operation_ids,
            .accepted_inputs = self.accepted_inputs,
        };
    }
};

pub const NormalizedModel = struct {
    descriptor: RuntimeModelDescriptor,
    artifacts: RuntimeArtifactSet,
    capabilities: RuntimeCapabilitySet,
    support: report_mod.RuntimeSupportReport,
    provider_key: types.ProviderKey,

    pub fn deinit(self: *NormalizedModel) void {
        self.descriptor.deinit();
        self.artifacts.deinit();
        self.support.deinit();
        self.* = undefined;
    }
};

fn pickTokenizerPath(
    allocator: std.mem.Allocator,
    catalog: *const catalog_mod.ArtifactCatalog,
) !?[]u8 {
    if (catalog.find(.tokenizer_json)) |item| return try allocator.dupe(u8, item.absolute_path);
    if (catalog.find(.tokenizer_model)) |item| return try allocator.dupe(u8, item.absolute_path);
    if (catalog.find(.vocab_json)) |item| return try allocator.dupe(u8, item.absolute_path);
    if (catalog.find(.vocab_txt)) |item| return try allocator.dupe(u8, item.absolute_path);
    return null;
}

fn freeOptional(allocator: std.mem.Allocator, value: ?[]u8) void {
    if (value) |owned| allocator.free(owned);
}
