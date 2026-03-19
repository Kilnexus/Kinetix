const std = @import("std");
const graph = @import("graph");
const ops = @import("ops");
const weights_mod = @import("weights");
const blocks = @import("blocks.zig");
const types = @import("../base/types.zig");
const utils = @import("../base/utils.zig");

pub const Tensor = types.Tensor;
pub const RuntimeError = types.RuntimeError;

pub fn runAttention(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "Attention")) return error.InvalidModuleKind;

    const num_heads: usize = @intCast(
        (module.getAttr("num_heads") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );
    if (num_heads == 0 or input.shape[1] % num_heads != 0) return error.InvalidAttributeType;

    var qkv_path_buffer: [256]u8 = undefined;
    const qkv_path = try utils.childModulePath(&qkv_path_buffer, module_path, "qkv");
    var proj_path_buffer: [256]u8 = undefined;
    const proj_path = try utils.childModulePath(&proj_path_buffer, module_path, "proj");
    var pe_path_buffer: [256]u8 = undefined;
    const pe_path = try utils.childModulePath(&pe_path_buffer, module_path, "pe");

    var qkv = try blocks.runConvModule(allocator, model_graph, weights_blob, qkv_path, input);
    defer qkv.deinit();

    const channels = input.shape[1];
    const head_dim = channels / num_heads;
    if (qkv.shape[1] < channels or (qkv.shape[1] - channels) % (2 * num_heads) != 0) return error.InvalidAttributeType;
    const key_dim = (qkv.shape[1] - channels) / (2 * num_heads);
    const per_head_span = 2 * key_dim + head_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(key_dim)));
    const spatial = input.shape[2] * input.shape[3];
    const batch_heads = input.shape[0] * num_heads;

    var attn = try allocator.alloc(f32, batch_heads * spatial * spatial);
    defer allocator.free(attn);
    @memset(attn, 0.0);

    for (0..input.shape[0]) |n| {
        for (0..num_heads) |head| {
            const attn_offset = (n * num_heads + head) * spatial * spatial;
            const attn_slice = attn[attn_offset .. attn_offset + spatial * spatial];

            for (0..spatial) |query_idx| {
                for (0..spatial) |key_idx| {
                    var acc: f32 = 0.0;
                    for (0..key_dim) |kd| {
                        const q_channel = head * per_head_span + kd;
                        const k_channel = head * per_head_span + key_dim + kd;
                        acc +=
                            qkv.data[(n * qkv.shape[1] + q_channel) * spatial + query_idx] *
                            qkv.data[(n * qkv.shape[1] + k_channel) * spatial + key_idx];
                    }
                    attn_slice[query_idx * spatial + key_idx] = acc * scale;
                }
            }
        }
    }
    try ops.softmaxRows(attn, batch_heads * spatial, spatial);

    var attended = try Tensor.init(allocator, input.shape[0], channels, input.shape[2], input.shape[3]);
    defer attended.deinit();
    attended.fill(0.0);

    for (0..input.shape[0]) |n| {
        for (0..num_heads) |head| {
            const attn_offset = (n * num_heads + head) * spatial * spatial;
            const attn_slice = attn[attn_offset .. attn_offset + spatial * spatial];

            for (0..head_dim) |hd| {
                const out_channel = head * head_dim + hd;
                const v_channel = head * per_head_span + 2 * key_dim + hd;
                for (0..spatial) |out_idx| {
                    var acc: f32 = 0.0;
                    for (0..spatial) |key_idx| {
                        acc +=
                            qkv.data[(n * qkv.shape[1] + v_channel) * spatial + key_idx] *
                            attn_slice[out_idx * spatial + key_idx];
                    }
                    attended.data[(n * channels + out_channel) * spatial + out_idx] = acc;
                }
            }
        }
    }

    var value_tensor = try Tensor.init(allocator, input.shape[0], channels, input.shape[2], input.shape[3]);
    defer value_tensor.deinit();
    for (0..input.shape[0]) |n| {
        for (0..num_heads) |head| {
            for (0..head_dim) |hd| {
                const dst_channel = head * head_dim + hd;
                const src_channel = head * per_head_span + 2 * key_dim + hd;
                for (0..spatial) |idx| {
                    value_tensor.data[(n * channels + dst_channel) * spatial + idx] =
                        qkv.data[(n * qkv.shape[1] + src_channel) * spatial + idx];
                }
            }
        }
    }

    var pe = try blocks.runConvModule(allocator, model_graph, weights_blob, pe_path, &value_tensor);
    defer pe.deinit();
    for (attended.data, pe.data) |*dst, src| dst.* += src;

    return try blocks.runConvModule(allocator, model_graph, weights_blob, proj_path, &attended);
}

pub fn runPSABlock(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "PSABlock")) return error.InvalidModuleKind;
    const add = (module.getAttr("add") orelse return error.MissingAttribute).asBool() orelse return error.InvalidAttributeType;

    var attn_path_buffer: [256]u8 = undefined;
    const attn_path = try utils.childModulePath(&attn_path_buffer, module_path, "attn");
    var attn_out = try runAttention(allocator, model_graph, weights_blob, attn_path, input);
    defer attn_out.deinit();
    if (add) {
        if (!attn_out.sameShape(input)) return ops.OpError.ShapeMismatch;
        for (attn_out.data, input.data) |*dst, src| dst.* += src;
    }

    var ffn_path_buffer: [256]u8 = undefined;
    const ffn_path = try utils.childModulePath(&ffn_path_buffer, module_path, "ffn");
    const ffn = model_graph.findModule(ffn_path) orelse return error.ModuleNotFound;

    var current = try attn_out.clone();
    defer current.deinit();

    for (ffn.children) |child| {
        const next = try blocks.runModule(allocator, model_graph, weights_blob, child.path, &current);
        current.deinit();
        current = next;
    }

    const output = try Tensor.init(allocator, current.shape[0], current.shape[1], current.shape[2], current.shape[3]);
    @memcpy(output.data, current.data);
    if (add) {
        for (output.data, attn_out.data) |*dst, src| dst.* += src;
    }
    return output;
}

pub fn runC2PSA(
    allocator: std.mem.Allocator,
    model_graph: *const graph.Graph,
    weights_blob: *const weights_mod.WeightsBlob,
    module_path: []const u8,
    input: *const Tensor,
) !Tensor {
    const module = model_graph.findModule(module_path) orelse return error.ModuleNotFound;
    if (!std.mem.eql(u8, module.kind, "C2PSA")) return error.InvalidModuleKind;

    const hidden_channels: usize = @intCast(
        (module.getAttr("c") orelse return error.MissingAttribute).asInteger() orelse return error.InvalidAttributeType,
    );

    var cv1_buffer: [256]u8 = undefined;
    const cv1_path = try utils.childModulePath(&cv1_buffer, module_path, "cv1");
    var cv2_buffer: [256]u8 = undefined;
    const cv2_path = try utils.childModulePath(&cv2_buffer, module_path, "cv2");
    var seq_buffer: [256]u8 = undefined;
    const seq_path = try utils.childModulePath(&seq_buffer, module_path, "m");

    var stem = try blocks.runConvModule(allocator, model_graph, weights_blob, cv1_path, input);
    defer stem.deinit();
    if (stem.shape[1] != hidden_channels * 2) return ops.OpError.ShapeMismatch;

    var right = try utils.sliceChannels(allocator, &stem, hidden_channels, hidden_channels);
    defer right.deinit();

    const seq = model_graph.findModule(seq_path) orelse return error.ModuleNotFound;
    for (seq.children) |child| {
        const next = try blocks.runModule(allocator, model_graph, weights_blob, child.path, &right);
        right.deinit();
        right = next;
    }

    var concat = try Tensor.init(allocator, stem.shape[0], hidden_channels * 2, stem.shape[2], stem.shape[3]);
    defer concat.deinit();
    try ops.copyChannelRange(&stem, 0, hidden_channels, &concat, 0);
    try ops.copyChannelRange(&right, 0, hidden_channels, &concat, hidden_channels);

    return try blocks.runConvModule(allocator, model_graph, weights_blob, cv2_path, &concat);
}
