"""Level 1: Pooling layers.

Shrink the image by summarizing small regions—keeps important information
while reducing the amount of data. Max and average pooling for spatial
downsampling.
"""

from dataclasses import dataclass

from policywerk.primitives import scalar
from policywerk.primitives.matrix import tensor3d_zeros

Tensor3D = list[list[list[float]]]


@dataclass
class MaxPoolCache:
    """Saved during forward pass for backprop—which input was the max."""
    in_channels: int
    in_height: int
    in_width: int
    # Remembers which input pixel was the maximum, so the backward pass knows where to route the gradient.
    max_indices: list[tuple[int, int, int, int, int]]  # (ch, out_r, out_c, in_r, in_c)


@dataclass
class AvgPoolCache:
    """Saved during forward pass for backprop—input dimensions needed to reconstruct gradient shape."""
    in_channels: int
    in_height: int
    in_width: int


def max_pool_forward(
    inputs: Tensor3D,
    pool_size: int = 2,   # size of the window to summarize (e.g. 2x2)
    stride: int = 2,      # how far to move the window each step (usually same as pool_size)
) -> tuple[Tensor3D, MaxPoolCache]:
    """Max pooling forward pass—take the maximum in each window."""
    channels = len(inputs)
    in_h = len(inputs[0])
    in_w = len(inputs[0][0])
    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1

    output = tensor3d_zeros(channels, out_h, out_w)
    max_indices = []

    for ch in range(channels):
        for out_row in range(out_h):
            for out_col in range(out_w):
                best = float("-inf")
                best_r, best_c = 0, 0
                for pr in range(pool_size):
                    for pc in range(pool_size):
                        r = out_row * stride + pr
                        c = out_col * stride + pc
                        if inputs[ch][r][c] > best:
                            best = inputs[ch][r][c]
                            best_r, best_c = r, c
                output[ch][out_row][out_col] = best
                max_indices.append((ch, out_row, out_col, best_r, best_c))

    cache = MaxPoolCache(
        in_channels=channels, in_height=in_h, in_width=in_w,
        max_indices=max_indices,
    )
    return output, cache


def max_pool_backward(
    output_grad: Tensor3D, cache: MaxPoolCache,
) -> Tensor3D:
    """Max pooling backward—gradient flows only to the max element."""
    input_grad = tensor3d_zeros(cache.in_channels, cache.in_height, cache.in_width)
    for ch, out_r, out_c, in_r, in_c in cache.max_indices:
        input_grad[ch][in_r][in_c] = scalar.add(
            input_grad[ch][in_r][in_c],
            output_grad[ch][out_r][out_c],
        )
    return input_grad


def avg_pool_forward(
    inputs: Tensor3D, pool_size: int = 2, stride: int = 2
) -> tuple[Tensor3D, AvgPoolCache]:
    """Average pooling forward pass."""
    channels = len(inputs)
    in_h = len(inputs[0])
    in_w = len(inputs[0][0])
    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1
    area = float(pool_size * pool_size)

    output = tensor3d_zeros(channels, out_h, out_w)

    for ch in range(channels):
        for out_row in range(out_h):
            for out_col in range(out_w):
                total = 0.0
                for pr in range(pool_size):
                    for pc in range(pool_size):
                        total = scalar.add(
                            total, inputs[ch][out_row * stride + pr][out_col * stride + pc]
                        )
                output[ch][out_row][out_col] = scalar.multiply(total, scalar.inverse(area))

    cache = AvgPoolCache(in_channels=channels, in_height=in_h, in_width=in_w)
    return output, cache


def avg_pool_backward(
    output_grad: Tensor3D, cache: AvgPoolCache,
    pool_size: int = 2, stride: int = 2,
) -> Tensor3D:
    """Average pooling backward—distribute gradient equally."""
    area = float(pool_size * pool_size)
    out_h = len(output_grad[0])
    out_w = len(output_grad[0][0])

    input_grad = tensor3d_zeros(cache.in_channels, cache.in_height, cache.in_width)

    for ch in range(cache.in_channels):
        for out_row in range(out_h):
            for out_col in range(out_w):
                distributed = scalar.multiply(
                    output_grad[ch][out_row][out_col], scalar.inverse(area)
                )
                for pr in range(pool_size):
                    for pc in range(pool_size):
                        in_row = out_row * stride + pr
                        in_col = out_col * stride + pc
                        input_grad[ch][in_row][in_col] = scalar.add(
                            input_grad[ch][in_row][in_col], distributed
                        )

    return input_grad
