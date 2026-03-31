test {
    _ = @import("graph");
    _ = @import("tensor");
    _ = @import("ops");
    _ = @import("runtime");
    _ = @import("weights");
    _ = @import("Pixio");
    _ = @import("imaging/imaging_tests.zig");
    _ = @import("vision/preprocess_tests.zig");
    _ = @import("runtime/runtime_tests.zig");
}
