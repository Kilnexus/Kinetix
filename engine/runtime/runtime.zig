pub const load_plan = @import("load_plan.zig");
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
pub const providers = struct {
    pub const chandra_native = @import("providers/chandra_native.zig");
    pub const chandra_preprocess = @import("providers/chandra_preprocess.zig");
    pub const chandra_store = @import("providers/chandra_store.zig");
    pub const chandra_vision = @import("providers/chandra_vision.zig");
    pub const chandra_weights = @import("providers/chandra_weights.zig");
    pub const swiftocr_native = @import("providers/swiftocr_native.zig");
    pub const text_shared = @import("providers/text_shared.zig");
    pub const vision_shared = @import("providers/vision_shared.zig");
};
