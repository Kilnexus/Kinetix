pub const Scheme = enum {
    auto,
    bf16,
    q6,
    q8,
    q4,

    pub fn name(self: Scheme) []const u8 {
        return switch (self) {
            .auto => "auto",
            .bf16 => "bf16",
            .q6 => "q6",
            .q8 => "q8",
            .q4 => "q4",
        };
    }
};
