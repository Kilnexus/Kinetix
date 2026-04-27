pub const load_plan = @import("load_plan.zig");
pub const abi = @import("runtime_abi");
pub const ocr_pipeline = @import("ocr_pipeline.zig");
pub const text = @import("text/text.zig");
pub const types = @import("types.zig");
pub const backend = @import("backend/registry.zig");
pub const catalog = @import("catalog/catalog.zig");
pub const model_resolver = @import("model/resolver/resolver.zig");
pub const model = struct {
    pub const RuntimeModelDescriptor = @import("model/descriptor.zig").RuntimeModelDescriptor;
    pub const RuntimeCapabilitySet = @import("model/features.zig").RuntimeCapabilitySet;
    pub const ModelHandle = @import("model/handle.zig").ModelHandle;
};
pub const registry = struct {
    pub const provider_registry = @import("registry/provider_registry.zig");
};
pub const planner = @import("planner/planner.zig");
pub const executor = @import("executor/executor.zig");
pub const session = @import("session/session.zig");
pub const families = struct {
    pub const registry = @import("families/registry.zig");
    pub const text = struct {
        pub const bert = @import("families/text/bert/family.zig");
        pub const qwen3 = @import("families/text/qwen3/family.zig");
    };
    pub const vision = struct {
        pub const yolo = @import("families/vision/yolo/family.zig");
    };
    pub const ocr = struct {
        pub const chandra = @import("families/ocr/chandra/family.zig");
        pub const swiftocr = @import("families/ocr/swiftocr/family.zig");
    };
    pub const tts = struct {
        pub const moss_tts_nano = @import("families/tts/moss_tts_nano/family.zig");
    };
    pub const generic = @import("families/generic/family.zig");
};
pub const shared = struct {
    pub const ops = @import("shared_ops");
    pub const text = @import("shared/text/runtime.zig");
    pub const vision = @import("shared/vision/runtime.zig");
    pub const ocr = @import("shared/ocr/runtime.zig");
};
pub const providers = struct {
    pub const chandra_native = families.ocr.chandra.native;
    pub const chandra_preprocess = families.ocr.chandra.preprocess;
    pub const chandra_store = families.ocr.chandra.store;
    pub const chandra_vision = families.ocr.chandra.vision;
    pub const chandra_weights = families.ocr.chandra.weights;
    pub const swiftocr_native = families.ocr.swiftocr.native;
    pub const text_shared = shared.text;
    pub const vision_shared = shared.vision;
    pub const ocr_shared = shared.ocr;
};
