const types = @import("types.zig");

pub const Tensor = types.Tensor;
pub const OpError = types.OpError;

pub fn siluInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = x / (1.0 + @exp(-x));
    }
}

pub fn sigmoidInPlace(tensor: *Tensor) void {
    for (tensor.data) |*value| {
        const x = value.*;
        value.* = 1.0 / (1.0 + @exp(-x));
    }
}

pub fn add(output: *Tensor, lhs: *const Tensor, rhs: *const Tensor) OpError!void {
    if (!output.sameShape(lhs) or !lhs.sameShape(rhs)) return OpError.ShapeMismatch;
    for (output.data, lhs.data, rhs.data) |*out, left, right| out.* = left + right;
}
