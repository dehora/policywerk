"""Level 1: Gradient computation.

Backpropagation logic for dense layers, plus numerical gradient
checking via finite differences.
"""

from dataclasses import dataclass

from policywerk.primitives import scalar, vector, matrix
from policywerk.primitives.activations import (
    sigmoid, sigmoid_derivative,
    tanh_, tanh_derivative,
    relu, relu_derivative,
    elu, elu_derivative,
    silu, silu_derivative,
    identity, identity_derivative,
)

Vector = list[float]
Matrix = list[list[float]]


_DERIVATIVES = {
    sigmoid: sigmoid_derivative,
    tanh_: tanh_derivative,
    relu: relu_derivative,
    elu: elu_derivative,
    silu: silu_derivative,
    identity: identity_derivative,
}


@dataclass
class LayerGradients:
    """Gradients for a single dense layer."""
    weight_grads: Matrix
    bias_grads: Vector


def backward(network, cache, loss_grad: Vector) -> list[LayerGradients]:
    """Compute gradients for every layer via backpropagation."""
    gradients = []
    delta = loss_grad

    for layer_idx in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_idx]
        layer_cache = cache.layer_caches[layer_idx]
        activation_fn = network.activation_fns[layer_idx]
        deriv_fn = _DERIVATIVES[activation_fn]

        f_prime = vector.apply(deriv_fn, layer_cache.z)
        delta = vector.elementwise(scalar.multiply, delta, f_prime)

        weight_grads = matrix.outer(delta, layer_cache.inputs)
        bias_grads = list(delta)

        gradients.append(LayerGradients(
            weight_grads=weight_grads,
            bias_grads=bias_grads,
        ))

        delta = matrix.mat_vec(matrix.transpose(layer.weights), delta)

    gradients.reverse()
    return gradients


def numerical_gradient_check(
    network,
    inputs: Vector,
    targets: Vector,
    loss_fn,
    epsilon: float = 1e-5,
) -> float:
    """Check analytical gradients against numerical (finite-difference) gradients.

    Returns the maximum relative error across all weights.
    """
    from policywerk.building_blocks.network import network_forward

    loss_derivative = _get_loss_derivative(loss_fn)
    output, cache = network_forward(network, inputs)
    loss_grad = loss_derivative(output, targets)
    analytical = backward(network, cache, loss_grad)

    max_error = 0.0

    for layer_idx in range(len(network.layers)):
        layer = network.layers[layer_idx]
        grads = analytical[layer_idx]

        for row in range(len(layer.weights)):
            for col in range(len(layer.weights[0])):
                original = layer.weights[row][col]

                layer.weights[row][col] = original + epsilon
                out_plus, _ = network_forward(network, inputs)
                loss_plus = loss_fn(out_plus, targets)

                layer.weights[row][col] = original - epsilon
                out_minus, _ = network_forward(network, inputs)
                loss_minus = loss_fn(out_minus, targets)

                layer.weights[row][col] = original

                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                analytical_grad = grads.weight_grads[row][col]

                denom = max(abs(numerical), abs(analytical_grad), 1e-8)
                error = abs(numerical - analytical_grad) / denom
                if error > max_error:
                    max_error = error

        for bias_idx in range(len(layer.biases)):
            original = layer.biases[bias_idx]

            layer.biases[bias_idx] = original + epsilon
            out_plus, _ = network_forward(network, inputs)
            loss_plus = loss_fn(out_plus, targets)

            layer.biases[bias_idx] = original - epsilon
            out_minus, _ = network_forward(network, inputs)
            loss_minus = loss_fn(out_minus, targets)

            layer.biases[bias_idx] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical_grad = grads.bias_grads[bias_idx]

            denom = max(abs(numerical), abs(analytical_grad), 1e-8)
            error = abs(numerical - analytical_grad) / denom
            if error > max_error:
                max_error = error

    return max_error


def _get_loss_derivative(loss_fn):
    """Look up the derivative for a loss function."""
    from policywerk.primitives.losses import (
        mse, mse_derivative,
        cross_entropy, cross_entropy_derivative,
        huber, huber_derivative,
    )
    if loss_fn is mse:
        return mse_derivative
    if loss_fn is cross_entropy:
        return cross_entropy_derivative
    if loss_fn is huber:
        return huber_derivative
    raise ValueError(f"Unknown loss function: {loss_fn}")
