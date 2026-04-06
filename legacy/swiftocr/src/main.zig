const std = @import("std");
const command = @import("cli/command.zig");
const pipeline_mod = @import("ocr/pipeline.zig");

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa_state.deinit() == .ok);
    const gpa = gpa_state.allocator();

    var stdout_writer = std.fs.File.stdout().writer(&.{});
    const out = &stdout_writer.interface;

    const parsed = try command.parse(gpa);
    defer parsed.deinit(gpa);

    if (parsed.help) {
        try command.printUsage(out);
        try out.flush();
        return;
    }

    switch (parsed.kind) {
        .infer => {
            const infer = parsed.infer.?;
            var pipeline = pipeline_mod.Pipeline.init(gpa);
            defer pipeline.deinit();

            const result = try pipeline.infer(.{
                .model_path = infer.model_path,
                .image_path = infer.image_path,
            });

            try out.print(
                "Inference completed.\nLoaded tensors: {d}\nImage: {d}x{d}\n",
                .{ result.loaded_tensors, result.image_width, result.image_height },
            );
            try out.flush();
        },
    }
}
