const std = @import("std");

pub const version_major: u16 = 1;
pub const version_minor: u16 = 0;
pub const version_patch: u16 = 0;

pub const Version = struct {
    major: u16 = version_major,
    minor: u16 = version_minor,
    patch: u16 = version_patch,

    pub fn compatibleWith(self: Version, other: Version) bool {
        return self.major == other.major and self.minor >= other.minor;
    }
};

pub const current_version = Version{};

pub const Operation = enum {
    infer,
    generate,
    chat,
    embed,
    detect,
    ocr,
    render_markdown,
    synthesize,
    fill_mask,
    profile,
    benchmark,

    pub fn name(self: Operation) []const u8 {
        return switch (self) {
            .infer => "infer",
            .generate => "generate",
            .chat => "chat",
            .embed => "embed",
            .detect => "detect",
            .ocr => "infer-ocr",
            .render_markdown => "render-markdown",
            .synthesize => "synthesize",
            .fill_mask => "fill-mask",
            .profile => "profile",
            .benchmark => "benchmark",
        };
    }

    pub fn parse(name_value: []const u8) ?Operation {
        inline for (std.meta.fields(Operation)) |field| {
            const value: Operation = @enumFromInt(field.value);
            if (std.mem.eql(u8, value.name(), name_value)) return value;
        }
        if (std.mem.eql(u8, name_value, "ocr")) return .ocr;
        return null;
    }
};

pub const InputKind = enum {
    none,
    text,
    image_path,
    document_path,
    audio_path,
    video_path,
};

pub const ExecutionPath = enum {
    runtime_backend,
    stream,
    native_accelerated,
    fallback,
};

pub const ExecutionOrigin = enum {
    runtime_backend,
    native_single,
    native_batch,
};

pub const BatchExecutionPath = enum {
    backend_batch,
    per_request_fallback,
};

pub const CapabilityFlags = packed struct {
    sync: bool = true,
    async_exec: bool = false,
    stream: bool = false,
    batch: bool = false,
    native_exec: bool = false,
};

pub const CapabilitySet = struct {
    flags: CapabilityFlags = .{},
    operations: []const Operation = &.{},
    accepted_inputs: []const InputKind = &.{},

    pub fn supportsOperation(self: CapabilitySet, operation: Operation) bool {
        if (self.operations.len == 0) return true;
        for (self.operations) |supported| {
            if (supported == operation) return true;
        }
        return false;
    }

    pub fn acceptsInput(self: CapabilitySet, input: InputKind) bool {
        if (input == .none) return true;
        if (self.accepted_inputs.len == 0) return true;
        for (self.accepted_inputs) |accepted| {
            if (accepted == input) return true;
        }
        return false;
    }
};

pub const ScalarType = enum {
    f32,
    f16,
    bf16,
    i8,
    u8,
    q8,
    q6,
    q4,
};

pub const TensorLayout = enum {
    contiguous,
    row_major,
    nchw,
    nhwc,
    token_major,
    head_major,
    paged_head_major,
};

pub const MemorySpace = enum {
    host,
    device,
    mapped,
};

pub const TensorAbi = struct {
    scalar: ScalarType,
    layout: TensorLayout = .contiguous,
    memory: MemorySpace = .host,
    rank: u8,
};

pub const KernelClass = enum {
    graph_op,
    activation,
    attention,
    conv,
    layout,
    linalg,
    normalization,
    pooling,
    quantized,
    vision,
};

pub const KernelAbi = struct {
    class: KernelClass,
    input: TensorAbi,
    output: TensorAbi,
    specialized: bool = false,
};

test "runtime ABI operation parser preserves public names" {
    try std.testing.expectEqual(Operation.generate, Operation.parse("generate").?);
    try std.testing.expectEqual(Operation.ocr, Operation.parse("infer-ocr").?);
    try std.testing.expectEqual(Operation.render_markdown, Operation.parse("render-markdown").?);
    try std.testing.expect(Operation.parse("unknown-op") == null);
}

test "runtime ABI version uses major/minor compatibility" {
    try std.testing.expect(current_version.compatibleWith(.{ .major = 1, .minor = 0, .patch = 0 }));
    try std.testing.expect(!current_version.compatibleWith(.{ .major = 2, .minor = 0, .patch = 0 }));
}
