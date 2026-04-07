const std = @import("std");
pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();

    const gpa = gpa_state.allocator();
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    const stdout = std.fs.File.stdout().deprecatedWriter();
    const show_help = args.len <= 1 or
        std.mem.eql(u8, args[1], "--help") or
        std.mem.eql(u8, args[1], "-h");

    try stdout.writeAll(
        \\Legacy Zinfer CLI has been retired in this monorepo.
        \\Use the unified Kinetix entrypoint instead.
        \\Example:
        \\  zig build run -- run --model-dir .\models\text\Qwen3-0.6B --operation generate --input "Hello"
        \\
    );

    if (show_help) return;
    return error.InvalidCommand;
}

test {
    std.testing.refAllDecls(@This());
    _ = @import("../../../../engine/runtime/text/core/cpu.zig");
    _ = @import("../../../../engine/runtime/text/chat_types.zig");
    _ = @import("../../../../engine/runtime/text/decoder_types.zig");
    _ = @import("../../../../engine/runtime/text/logits.zig");
    _ = @import("../../../../engine/runtime/text/weights_layout.zig");
    _ = @import("../../../../engine/runtime/text/block_layout.zig");
    _ = @import("../../../../engine/runtime/text/gqa_attention.zig");
    _ = @import("../../../../engine/runtime/text/attention/attention.zig");
    _ = @import("../../../../engine/runtime/text/tensor/bfloat16.zig");
    _ = @import("../../../../engine/runtime/text/kv_cache/quantize.zig");
    _ = @import("../../../../engine/runtime/text/kv_cache/cache.zig");
    _ = @import("../../../../engine/runtime/text/optimized_decoder/runtime.zig");
    _ = @import("../../../../engine/runtime/text/optimized_decoder/batch.zig");
    _ = @import("../../../../engine/runtime/text/optimized_decoder/workspace.zig");
    _ = @import("../../../../engine/runtime/text/backend/backend.zig");
    _ = @import("../../../../engine/runtime/text/bpe.zig");
    _ = @import("../../../../engine/runtime/text/quantized.zig");
    _ = @import("sampling/sampler.zig");
}
