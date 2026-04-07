pub const ocr = struct {
    pub const Pipeline = @import("ocr/pipeline.zig").Pipeline;
};

pub const artifacts = struct {
    pub const Model = @import("engine_ocr_model").Model;
    pub const TensorBlob = @import("engine_ocr_model").TensorBlob;
    pub const Image = @import("engine_ocr_image").Image;
};

pub const memory = struct {
    pub const ArenaPool = @import("engine_arena_pool").ArenaPool;
};

test {
    _ = @import("engine_arena_pool");
    _ = @import("engine_ocr_model");
    _ = @import("engine_ocr_image");
    _ = @import("ocr/pipeline.zig");
}
