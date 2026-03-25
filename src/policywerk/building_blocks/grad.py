"""Level 1: Gradient computation (backpropagation).

Backpropagation is the algorithm that makes neural network training
possible. Before Rumelhart, Hinton, and Williams published it in
1986, there was no efficient way to train networks with hidden
layers—you could build them, but you couldn't teach them.

The problem backpropagation solves is credit assignment in a network:
when the output is wrong, which weights are responsible? A network
might have thousands of weights across many layers, and the output
error is a combined result of all of them. Adjusting weights randomly
would take forever. Backpropagation finds the answer in a single
backward pass.

The key insight is the chain rule from calculus. If the network
computes y = f(g(h(x))), then the sensitivity of y to a small
change in x flows backward through each function:

    dy/dx = dy/df × df/dg × dg/dh × dh/dx

In a network, each layer is one of those functions. The backward
pass starts at the loss (how wrong was the output?) and works
backward through each layer, asking two questions:

  1. How much did each weight in this layer contribute to the error?
     → This gives the weight gradients (used by the optimizer).

  2. How much error should be passed to the previous layer?
     → This gives the input gradient (used by the next layer back).

The forward pass cached intermediate values at each layer (the
inputs, pre-activation sums, and post-activation outputs). The
backward pass uses those cached values to compute gradients
without re-running the forward computation.

Concretely, at each layer the backward pass computes:

    delta    = incoming_error × activation_derivative(pre_activation)
    dW       = outer_product(delta, layer_inputs)
    db       = delta
    pass_back = transpose(weights) × delta

This module also includes numerical gradient checking—a slow but
reliable way to verify that the analytical gradients from backprop
are correct, by wiggling each weight and measuring the effect.
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


# Maps each activation function to its derivative—during backprop, we need
# the derivative of whatever activation was used in the forward pass.
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
    """Compute gradients for every layer via backpropagation.

    Walk backward through the network. At each layer:
    (1) compute how sensitive the activation was to its input (f_prime),
    (2) multiply by the incoming error signal to get delta,
    (3) compute weight gradients from delta and the layer's inputs,
    (4) pass the error signal to the previous layer through the transposed weights.
    """
    gradients = []
    # delta is the error signal flowing backward—starts as 'how wrong was the output?'
    delta = loss_grad

    for layer_idx in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[layer_idx]
        layer_cache = cache.layer_caches[layer_idx]
        activation_fn = network.activation_fns[layer_idx]
        deriv_fn = _DERIVATIVES[activation_fn]

        # how sensitive each activation was to its pre-activation input
        f_prime = vector.apply(deriv_fn, layer_cache.z)
        delta = vector.elementwise(scalar.multiply, delta, f_prime)

        # each weight's gradient = error at this neuron × input that fed it
        weight_grads = matrix.outer(delta, layer_cache.inputs)
        bias_grads = list(delta)

        gradients.append(LayerGradients(
            weight_grads=weight_grads,
            bias_grads=bias_grads,
        ))

        # pass error signal backward through this layer's weights to the previous layer
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

    Wiggle each weight by a tiny amount (+/- epsilon) and measure how the loss
    changes. This gives a slow but reliable gradient that we compare against the
    fast analytical gradient from backward(). If they match, the backward pass
    is correct.

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
