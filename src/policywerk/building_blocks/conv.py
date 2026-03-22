"""Level 1: Convolutional layer.

Slides small patterns (filters) across an image, checking at each position how
well the pattern matches. Each filter learns to detect a different feature — an
edge, a corner, a texture.

2D cross-correlation with learnable filters.
Used by DQN for spatial feature extraction from game frames.
"""

from dataclasses import dataclass

from policywerk.primitives import scalar
from policywerk.primitives.matrix import tensor3d_zeros

Tensor3D = list[list[list[float]]]
Vector = list[float]


@dataclass
class ConvLayer:
    filters: list[Tensor3D]  # num_filters x in_channels x kH x kW
    biases: Vector            # num_filters


@dataclass
class ConvCache:
    """Values saved during forward pass, needed for backprop."""
    inputs: Tensor3D
    z: Tensor3D
    a: Tensor3D


def create_conv(rng, in_channels: int, out_channels: int, kernel_size: int = 3) -> ConvLayer:
    """Create a conv layer with Xavier-initialized filters and zero biases.

    kernel_size is the width and height of each filter (also called 'kernel').
    in_channels = how many layers the input has (e.g. 1 for grayscale, 4 for
    stacked game frames). out_channels = how many different features to detect.
    """
    fan_in = in_channels * kernel_size * kernel_size
    fan_out = out_channels * kernel_size * kernel_size
    limit = (6.0 / (fan_in + fan_out)) ** 0.5

    filters = []
    for _ in range(out_channels):
        f = []
        for _ in range(in_channels):
            channel = []
            for _ in range(kernel_size):
                row = [rng.uniform(-limit, limit) for _ in range(kernel_size)]
                channel.append(row)
            f.append(channel)
        filters.append(f)

    biases = [0.0] * out_channels
    return ConvLayer(filters=filters, biases=biases)


def conv_forward(
    layer: ConvLayer, inputs: Tensor3D, activation_fn
) -> tuple[Tensor3D, ConvCache]:
    """2D cross-correlation forward pass."""
    in_channels = len(inputs)
    in_h = len(inputs[0])
    in_w = len(inputs[0][0])
    num_filters = len(layer.filters)
    k = len(layer.filters[0][0])
    out_h = in_h - k + 1
    out_w = in_w - k + 1

    z = tensor3d_zeros(num_filters, out_h, out_w)
    a = tensor3d_zeros(num_filters, out_h, out_w)

    # For each filter, at each output position, compute the dot product
    # between the filter and the corresponding input patch.
    for filt in range(num_filters):
        for out_row in range(out_h):
            for out_col in range(out_w):
                val = layer.biases[filt]
                for ch in range(in_channels):
                    for kh in range(k):
                        for kw in range(k):
                            val = scalar.add(
                                val,
                                scalar.multiply(
                                    layer.filters[filt][ch][kh][kw],
                                    inputs[ch][out_row + kh][out_col + kw],
                                ),
                            )
                z[filt][out_row][out_col] = val
                a[filt][out_row][out_col] = activation_fn(val)

    cache = ConvCache(inputs=inputs, z=z, a=a)
    return a, cache


def conv_backward(
    layer: ConvLayer,
    cache: ConvCache,
    output_grad: Tensor3D,
    deriv_fn,
) -> tuple[Tensor3D, list[Tensor3D], Vector]:
    """Backward pass for conv layer.

    Returns (input_grad, filter_grads, bias_grads).
    """
    num_filters = len(layer.filters)
    in_channels = len(cache.inputs)
    in_h = len(cache.inputs[0])
    in_w = len(cache.inputs[0][0])
    k = len(layer.filters[0][0])
    out_h = len(output_grad[0])
    out_w = len(output_grad[0][0])

    # Step 1: compute delta — error signal adjusted for the activation
    delta = tensor3d_zeros(num_filters, out_h, out_w)
    for filt in range(num_filters):
        for out_row in range(out_h):
            for out_col in range(out_w):
                delta[filt][out_row][out_col] = scalar.multiply(
                    output_grad[filt][out_row][out_col],
                    deriv_fn(cache.z[filt][out_row][out_col]),
                )

    # Step 2: bias gradients — sum delta over all spatial positions
    bias_grads = [0.0] * num_filters
    for filt in range(num_filters):
        for out_row in range(out_h):
            for out_col in range(out_w):
                bias_grads[filt] = scalar.add(bias_grads[filt], delta[filt][out_row][out_col])

    # Step 3: filter gradients — how much should each filter weight change?
    filter_grads = []
    for filt in range(num_filters):
        fg = []
        for ch in range(in_channels):
            channel = []
            for kh in range(k):
                row = []
                for kw in range(k):
                    grad = 0.0
                    for out_row in range(out_h):
                        for out_col in range(out_w):
                            grad = scalar.add(
                                grad,
                                scalar.multiply(
                                    delta[filt][out_row][out_col],
                                    cache.inputs[ch][out_row + kh][out_col + kw],
                                ),
                            )
                    row.append(grad)
                channel.append(row)
            fg.append(channel)
        filter_grads.append(fg)

    # Step 4: input gradients — how much should each input pixel change?
    input_grad = tensor3d_zeros(in_channels, in_h, in_w)
    for ch in range(in_channels):
        for in_row in range(in_h):
            for in_col in range(in_w):
                val = 0.0
                for filt in range(num_filters):
                    for kh in range(k):
                        for kw in range(k):
                            oi = in_row - kh
                            oj = in_col - kw
                            if 0 <= oi < out_h and 0 <= oj < out_w:
                                val = scalar.add(
                                    val,
                                    scalar.multiply(
                                        delta[filt][oi][oj],
                                        layer.filters[filt][ch][kh][kw],
                                    ),
                                )
                input_grad[ch][in_row][in_col] = val

    return input_grad, filter_grads, bias_grads
