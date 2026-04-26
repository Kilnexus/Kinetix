pub const q8_cache_group_size: usize = 16;
pub const handwritten_q8_head_dim: usize = 128;
pub const handwritten_q8_scale_groups: usize = handwritten_q8_head_dim / q8_cache_group_size;
pub const paired_scores_max_seq_len: usize = 4096;
