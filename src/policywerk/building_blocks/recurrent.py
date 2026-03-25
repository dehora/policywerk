"""Level 1: Recurrent layers.

GRU (Gated Recurrent Unit) for sequence modeling.
Used by DreamerV3's RSSM world model to maintain temporal state.

A GRU processes sequences one step at a time, maintaining a hidden
state h that summarizes history. Two gates control information flow:

    z (update gate): how much of the old state to keep
    r (reset gate): how much of the old state to expose when computing new content

    z = σ(W_z @ [h, x] + b_z)
    r = σ(W_r @ [h, x] + b_r)
    h_candidate = tanh(W_h @ [r*h, x] + b_h)
    h_new = (1-z) * h + z * h_candidate
"""

from dataclasses import dataclass

from policywerk.primitives import scalar, vector, matrix
from policywerk.primitives.activations import sigmoid, tanh_
from policywerk.primitives.random import xavier_init

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class GRULayer:
    """GRU parameters—three sets of weights for update, reset, and candidate."""
    W_z: Matrix    # update gate weights (hidden_size, hidden_size + input_size)
    b_z: Vector    # update gate biases
    W_r: Matrix    # reset gate weights
    b_r: Vector    # reset gate biases
    W_h: Matrix    # candidate weights
    b_h: Vector    # candidate biases
    hidden_size: int


@dataclass
class GRUCache:
    """Values saved during forward pass for backprop."""
    h_prev: Vector       # previous hidden state
    x: Vector            # input
    z_gate: Vector       # update gate output
    r_gate: Vector       # reset gate output
    h_candidate: Vector  # candidate hidden state
    h_new: Vector        # new hidden state


def create_gru(rng, input_size: int, hidden_size: int) -> GRULayer:
    """Create a GRU layer with Xavier-initialized weights."""
    combined = hidden_size + input_size
    return GRULayer(
        W_z=xavier_init(rng, combined, hidden_size),
        b_z=vector.zeros(hidden_size),
        W_r=xavier_init(rng, combined, hidden_size),
        b_r=vector.zeros(hidden_size),
        W_h=xavier_init(rng, combined, hidden_size),
        b_h=vector.zeros(hidden_size),
        hidden_size=hidden_size,
    )


def gru_forward(
    layer: GRULayer, h_prev: Vector, x: Vector
) -> tuple[Vector, GRUCache]:
    """One GRU timestep: (h_prev, x) → h_new.

    Returns the new hidden state and cache for backprop.
    """
    combined = vector.concat(h_prev, x)

    # Update gate: how much to keep from old state
    z_pre = vector.add(matrix.mat_vec(layer.W_z, combined), layer.b_z)
    z_gate = vector.apply(sigmoid, z_pre)

    # Reset gate: how much of old state to expose for candidate
    r_pre = vector.add(matrix.mat_vec(layer.W_r, combined), layer.b_r)
    r_gate = vector.apply(sigmoid, r_pre)

    # Candidate: new content to potentially write
    reset_h = vector.elementwise(scalar.multiply, r_gate, h_prev)
    combined_reset = vector.concat(reset_h, x)
    h_pre = vector.add(matrix.mat_vec(layer.W_h, combined_reset), layer.b_h)
    h_candidate = vector.apply(tanh_, h_pre)

    # New hidden state: interpolate between old and candidate
    h_new = []
    for i in range(layer.hidden_size):
        h_new.append(scalar.add(
            scalar.multiply(scalar.subtract(1.0, z_gate[i]), h_prev[i]),
            scalar.multiply(z_gate[i], h_candidate[i]),
        ))

    cache = GRUCache(
        h_prev=h_prev, x=x,
        z_gate=z_gate, r_gate=r_gate,
        h_candidate=h_candidate, h_new=h_new,
    )
    return h_new, cache


def gru_backward(
    layer: GRULayer, cache: GRUCache, grad_h_new: Vector
) -> tuple[Vector, Vector, GRULayer]:
    """GRU backward pass.

    TODO: decompose into per-gate helper functions before L07 (DreamerV3).
    This 100-line function works but is hard to debug. Splitting into
    _backward_interpolation, _backward_candidate, _backward_reset_gate,
    _backward_update_gate would make it testable per-component.

    Reverse each forward step, computing how the error flows backward through:
    (1) the output interpolation, (2) the candidate computation, (3) the reset
    gate, (4) the update gate. Each path contributes gradients to the previous
    hidden state, the input, and the weights.

    Returns (grad_h_prev, grad_x, grad_layer) where grad_layer contains
    accumulated weight/bias gradients in a GRULayer-shaped structure.
    """
    hs = layer.hidden_size
    xs = len(cache.x)

    # Gradient through the interpolation: h_new = (1-z)*h_prev + z*h_candidate
    grad_h_prev_interp = [
        scalar.multiply(scalar.subtract(1.0, cache.z_gate[i]), grad_h_new[i])
        for i in range(hs)
    ]
    grad_h_candidate = [
        scalar.multiply(cache.z_gate[i], grad_h_new[i])
        for i in range(hs)
    ]
    grad_z = [
        scalar.multiply(
            grad_h_new[i],
            scalar.subtract(cache.h_candidate[i], cache.h_prev[i]),
        )
        for i in range(hs)
    ]

    # Gradient through h_candidate = tanh(W_h @ [r*h, x] + b_h)
    reset_h = vector.elementwise(scalar.multiply, cache.r_gate, cache.h_prev)
    combined_reset = vector.concat(reset_h, cache.x)
    h_pre = vector.add(matrix.mat_vec(layer.W_h, combined_reset), layer.b_h)
    grad_h_pre = [
        scalar.multiply(grad_h_candidate[i],
                        scalar.subtract(1.0, scalar.multiply(cache.h_candidate[i], cache.h_candidate[i])))
        for i in range(hs)
    ]

    grad_W_h = matrix.outer(grad_h_pre, combined_reset)
    grad_b_h = list(grad_h_pre)
    grad_combined_reset = matrix.mat_vec(matrix.transpose(layer.W_h), grad_h_pre)

    grad_reset_h = grad_combined_reset[:hs]
    grad_x_from_h = grad_combined_reset[hs:]

    # Gradient through reset: reset_h = r * h_prev
    grad_r = vector.elementwise(scalar.multiply, grad_reset_h, cache.h_prev)
    grad_h_prev_from_reset = vector.elementwise(scalar.multiply, grad_reset_h, cache.r_gate)

    # Gradient through r_gate = sigmoid(W_r @ [h, x] + b_r)
    combined = vector.concat(cache.h_prev, cache.x)
    r_pre = vector.add(matrix.mat_vec(layer.W_r, combined), layer.b_r)
    grad_r_pre = [
        scalar.multiply(grad_r[i],
                        scalar.multiply(cache.r_gate[i],
                                        scalar.subtract(1.0, cache.r_gate[i])))
        for i in range(hs)
    ]
    grad_W_r = matrix.outer(grad_r_pre, combined)
    grad_b_r = list(grad_r_pre)
    grad_combined_r = matrix.mat_vec(matrix.transpose(layer.W_r), grad_r_pre)

    # Gradient through z_gate = sigmoid(W_z @ [h, x] + b_z)
    z_pre = vector.add(matrix.mat_vec(layer.W_z, combined), layer.b_z)
    grad_z_pre = [
        scalar.multiply(grad_z[i],
                        scalar.multiply(cache.z_gate[i],
                                        scalar.subtract(1.0, cache.z_gate[i])))
        for i in range(hs)
    ]
    grad_W_z = matrix.outer(grad_z_pre, combined)
    grad_b_z = list(grad_z_pre)
    grad_combined_z = matrix.mat_vec(matrix.transpose(layer.W_z), grad_z_pre)

    # Accumulate gradients flowing back to h_prev and x
    grad_h_prev = vector.add(
        grad_h_prev_interp,
        vector.add(
            grad_h_prev_from_reset,
            vector.add(
                grad_combined_r[:hs],
                grad_combined_z[:hs],
            ),
        ),
    )
    grad_x = vector.add(
        grad_x_from_h,
        vector.add(
            grad_combined_r[hs:],
            grad_combined_z[hs:],
        ),
    )

    grad_layer = GRULayer(
        W_z=grad_W_z, b_z=grad_b_z,
        W_r=grad_W_r, b_r=grad_b_r,
        W_h=grad_W_h, b_h=grad_b_h,
        hidden_size=hs,
    )

    return grad_h_prev, grad_x, grad_layer
