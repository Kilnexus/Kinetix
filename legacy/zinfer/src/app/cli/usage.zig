const std = @import("std");

pub fn printUsage() !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.writeAll(
        \\Usage:
        \\  zinfer
        \\  zinfer quantize <q8|q6|q4>
        \\  zinfer quantize <q8|q6|q4> [model_dir]
        \\  zinfer tokenize <text>
        \\  zinfer tokenize [model_dir] <text>
        \\  zinfer decode-ids <ids_csv>
        \\  zinfer decode-ids [model_dir] <ids_csv>
        \\  zinfer fill-mask <text> [top_k]
        \\  zinfer fill-mask [model_dir] <text> [top_k]
        \\  zinfer embed-text <text> [cls|mean] [count]
        \\  zinfer embed-text [model_dir] <text> [cls|mean] [count]
        \\  zinfer serve-bert [port] [bind_host] [runtime_count]
        \\  zinfer serve-bert [model_dir] [port] [bind_host] [runtime_count]
        \\  zinfer generate <text> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate [model_dir] <text> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer generate-chat <messages_json_path> [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer generate-chat [model_dir] <messages_json_path> <max_new_tokens> [think|no-think] [flags...]
        \\  zinfer chat [max_new_tokens] [think|no-think] [flags...]
        \\  zinfer chat [model_dir] [max_new_tokens] [think|no-think] [flags...]
        \\
        \\Defaults:
        \\  model_dir = models/Qwen3-0.6B
        \\  generate max_new_tokens = 64
        \\  generate-chat/chat max_new_tokens = 128
        \\
        \\Flags:
        \\  --system <text>
        \\  --seed <u64>
        \\  --temperature <f32>
        \\  --top-p <f32>
        \\  --top-k <usize>
        \\  --min-p <f32>
        \\  --presence-penalty <f32>
        \\  --frequency-penalty <f32>
        \\  --repetition-penalty <f32>
        \\  --stop <text>           (repeatable)
        \\  --backend <auto|bf16|q8|q6|q4>
        \\  --kv-cache <auto|bf16|q8>
        \\  --q8-layout <token_major_legacy|head_major|paged_head_major>
        \\  --threads <usize>       (0 = auto)
        \\  --stream
        \\  --load <path>           (chat only)
        \\  --save <path>           (chat only)
        \\
    );
}
