const types = @import("types.zig");
const kernels = @import("shared_ops").kernels;

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;
const lane_count = 8;
const F32xN = @Vector(lane_count, f32);

pub fn siluInPlace(tensor: *Tensor) void {
    kernels.activation.siluInPlace(tensor.data);
}

pub fn sigmoidInPlace(tensor: *Tensor) void {
    kernels.activation.sigmoidInPlace(tensor.data);
}

pub fn add(output: *Tensor, lhs: *const Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(lhs) or !lhs.sameShape(rhs)) return OpError.ShapeMismatch;
    for (output.data, lhs.data, rhs.data) |*out, left, right| out.* = left + right;
}

pub fn addInPlace(output: *Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(rhs)) return OpError.ShapeMismatch;
    addInPlaceUnchecked(output, rhs);
}

pub fn addInPlaceUnchecked(output: *Tensor, rhs: *const Tensor) void {
    var i: usize = 0;
    while (i + lane_count <= output.data.len) : (i += lane_count) {
        const lhs_vec: F32xN = output.data[i..][0..lane_count].*;
        const rhs_vec: F32xN = rhs.data[i..][0..lane_count].*;
        output.data[i..][0..lane_count].* = lhs_vec + rhs_vec;
    }
    while (i < output.data.len) : (i += 1) {
        output.data[i] += rhs.data[i];
    }
}
