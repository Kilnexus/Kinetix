const std = @import("std");
const image_types = @import("../../types.zig");

pub const ImageU8 = image_types.ImageU8;

pub const WebpKind = enum {
    vp8,
    vp8l,
    vp8x,
};

pub const WebpChunkTag = enum {
    vp8,
    vp8l,
    vp8x,
    alph,
    anim,
    anmf,
    iccp,
    exif,
    xmp,
    unknown,
};

pub const WebpChunk = struct {
    tag: WebpChunkTag,
    payload: []const u8,
};

pub const Vp8lTransformType = enum {
    predictor,
    color,
    subtract_green,
    color_indexing,
};

pub const Vp8lImageRole = enum {
    argb,
    predictor,
    color,
    color_indexing,
    entropy,
};

pub const Vp8lPrefixCodeKind = enum {
    simple,
    normal,
};

pub const Vp8lSimplePrefixCode = struct {
    num_symbols: usize,
    is_first_8bits: bool,
    symbol0: usize,
    symbol1: ?usize,
    canonical_summary: ?Vp8lCanonicalPrefixSummary = null,
    end_bit_pos: usize,
};

pub const Vp8lNormalPrefixCode = struct {
    num_code_length_codes: usize,
    code_length_code_lengths: [19]usize,
    use_explicit_max_symbol: bool,
    length_nbits: ?usize,
    max_symbol: usize,
    decoded_symbol_tokens: ?usize = null,
    emitted_code_lengths: ?usize = null,
    non_zero_code_lengths: ?usize = null,
    preview_len: usize = 0,
    preview: [32]u8 = [_]u8{0} ** 32,
    canonical_summary: ?Vp8lCanonicalPrefixSummary = null,
    end_bit_pos: usize,
};

pub const Vp8lCanonicalCodeEntry = struct {
    symbol: usize,
    len: usize,
    lsb_code: usize,
};

pub const Vp8lCanonicalPrefixSummary = struct {
    active_symbol_count: usize,
    max_code_length: usize,
    preview_len: usize,
    preview: [16]Vp8lCanonicalCodeEntry,
};

pub const Vp8lCanonicalSymbolStream = struct {
    start_bit_pos: usize,
    end_bit_pos: usize,
    symbol_count: usize,
    preview_len: usize,
    preview: [32]usize,
};

pub const Vp8lPrefixCodeGroupDetail = struct {
    start_bit_pos: usize,
    end_bit_pos: usize,
    alphabet_sizes: [5]usize,
    group: Vp8lPrefixCodeGroup,
};

pub const Vp8lEventKind = enum {
    literal,
    copy,
    color_cache,
};

pub const Vp8lEvent = struct {
    kind: Vp8lEventKind,
    green: u16 = 0,
    red: u16 = 0,
    blue: u16 = 0,
    alpha: u16 = 0,
    cache_index: ?usize = null,
    length_symbol: ?usize = null,
    length: ?usize = null,
    distance_symbol: ?usize = null,
    distance_code: ?usize = null,
    distance: ?usize = null,
};

pub const Vp8lEventStream = struct {
    prefix_group_start_bit_pos: usize,
    event_stream_start_bit_pos: usize,
    end_bit_pos: usize,
    event_count: usize,
    emitted_pixels: usize,
    preview_len: usize,
    preview: [32]Vp8lEvent,
};

pub const Vp8lArgbImage = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    end_bit_pos: usize,
    pixels: []u32,

    pub fn deinit(self: *Vp8lArgbImage) void {
        self.allocator.free(self.pixels);
        self.* = undefined;
    }
};

pub const Vp8lPrefixCodeHeader = struct {
    kind: Vp8lPrefixCodeKind,
    start_bit_pos: usize,
    simple: ?Vp8lSimplePrefixCode = null,
    normal: ?Vp8lNormalPrefixCode = null,
};

pub const Vp8lPrefixCodeGroup = struct {
    parsed_count: usize,
    all_simple: bool,
    codes: [5]Vp8lPrefixCodeHeader,
};

pub const Vp8lEntropyImageDataHeader = struct {
    width: usize,
    height: usize,
    start_bit_pos: usize,
    use_color_cache: bool,
    color_cache_bits: ?usize,
    header_end_bit_pos: usize,
    prefix_codes_start_bit_pos: usize,
    prefix_group: Vp8lPrefixCodeGroup,
};

pub const Vp8lImageDataHeader = struct {
    role: Vp8lImageRole,
    width: usize,
    height: usize,
    start_bit_pos: usize,
    use_color_cache: bool,
    color_cache_bits: ?usize,
    meta_prefix_present: ?bool,
    prefix_bits: ?usize,
    prefix_image_width: ?usize,
    prefix_image_height: ?usize,
    prefix_image_start_bit_pos: ?usize,
    prefix_image_header: ?Vp8lEntropyImageDataHeader,
    header_end_bit_pos: usize,
    prefix_codes_start_bit_pos: ?usize,
    prefix_group: ?Vp8lPrefixCodeGroup,
};

pub const Vp8lTransform = struct {
    kind: Vp8lTransformType,
    size_bits: ?usize = null,
    color_table_size: ?usize = null,
    width_bits: ?usize = null,
    subimage_start_bit_pos: ?usize = null,
    subimage_width: ?usize = null,
    subimage_height: ?usize = null,
    subimage_header: ?Vp8lImageDataHeader = null,
    transform_width: ?usize = null,
    transform_height: ?usize = null,
    next_image_width: usize,
    next_image_height: usize,
};

pub const Vp8lStreamInfo = struct {
    width: usize,
    height: usize,
    has_alpha: bool,
    header_end_bit_pos: usize,
    image_data_start_bit_pos: ?usize,
    main_image_header: ?Vp8lImageDataHeader,
    transform_count: usize,
    transforms: [4]Vp8lTransform,
    tail_flags_known: bool,
    use_color_cache: ?bool,
    color_cache_bits: ?usize,
    use_meta_prefix: ?bool,
};

pub const WebpInfo = struct {
    width: usize,
    height: usize,
    has_alpha: bool,
    is_animated: bool,
    has_icc: bool,
    has_exif: bool,
    has_xmp: bool,
    kind: WebpKind,
};

pub const WebpError = image_types.ImageError || error{
    InvalidWebpHeader,
    InvalidWebpChunk,
    InvalidWebpData,
    MissingWebpChunk,
    TooManyWebpTransforms,
    UnsupportedWebpAnimation,
    UnsupportedWebpBitstream,
};
