const std = @import("std");

pub const SourceKind = enum {
    official_paddle_static,
    official_conversion_required,
};

pub const ModelSource = struct {
    stage: []const u8,
    variant: []const u8,
    format: []const u8,
    source_kind: SourceKind,
    url: []const u8,
};

pub const pp_ocrv5_sources = [_]ModelSource{
    .{
        .stage = "det",
        .variant = "PP-OCRv5_server",
        .format = "paddle_static",
        .source_kind = .official_paddle_static,
        .url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar",
    },
    .{
        .stage = "rec",
        .variant = "PP-OCRv5_server",
        .format = "paddle_static",
        .source_kind = .official_paddle_static,
        .url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar",
    },
    .{
        .stage = "det",
        .variant = "PP-OCRv5_mobile",
        .format = "paddle_static",
        .source_kind = .official_paddle_static,
        .url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar",
    },
    .{
        .stage = "rec",
        .variant = "PP-OCRv5_mobile",
        .format = "paddle_static",
        .source_kind = .official_paddle_static,
        .url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar",
    },
};

pub fn latestSources() []const ModelSource {
    return pp_ocrv5_sources[0..];
}

pub fn find(stage: []const u8, variant: []const u8) ?ModelSource {
    for (pp_ocrv5_sources) |source| {
        if (std.mem.eql(u8, source.stage, stage) and std.mem.eql(u8, source.variant, variant)) return source;
    }
    return null;
}

test "paddleocr model source catalog exposes pp-ocrv5 server det" {
    const source = find("det", "PP-OCRv5_server") orelse return error.ExpectedSource;
    try std.testing.expect(std.mem.endsWith(u8, source.url, "PP-OCRv5_server_det_infer.tar"));
}
