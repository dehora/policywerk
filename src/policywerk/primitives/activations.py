"""Level 0: Activation functions and normalization.

Non-linear functions applied after a neuron's weighted sum.
Without these, stacking layers would be pointless — multiple
linear transformations collapse into one. Activations introduce
curves and thresholds that let networks learn complex patterns.

Each activation includes its derivative (how sensitive the output
is to changes in the input). During training, the network uses
derivatives to figure out which direction to adjust each weight —
this is the core of backpropagation.
"""

import math

from policywerk.primitives import scalar, vector

Vector = list[float]


def step(x: float) -> float:
    """Return 1 if x >= 0, else 0 — the binary threshold activation."""
    return 1.0 if x >= 0 else 0.0


def sigmoid(x: float) -> float:
    """Squash x into (0, 1) — used to represent probabilities.

    Large positive x → near 1, large negative x → near 0,
    x = 0 → exactly 0.5. Formula: 1 / (1 + e^(-x))
    """
    return scalar.inverse(scalar.add(1.0, scalar.exp(scalar.negate(x))))


def sigmoid_derivative(x: float) -> float:
    """How fast sigmoid changes at x: σ(x) × (1 - σ(x))."""
    s = sigmoid(x)
    return scalar.multiply(s, scalar.subtract(1.0, s))


def tanh_(x: float) -> float:
    """Squash x into (-1, 1) — like sigmoid but centered at zero.

    Trailing underscore avoids shadowing math.tanh.
    Formula: (e^x - e^(-x)) / (e^x + e^(-x))
    """
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    """How fast tanh changes at x: 1 - tanh(x)²."""
    t = tanh_(x)
    return scalar.subtract(1.0, scalar.multiply(t, t))


def relu(x: float) -> float:
    """Return x if positive, else 0 — the rectified linear unit.

    Simple and effective: lets positive signals through unchanged,
    blocks negative ones entirely.
    """
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return 1.0 if x > 0 else 0.0


def silu(x: float) -> float:
    """SiLU (Sigmoid Linear Unit): x × sigmoid(x).

    Smooth activation that can dip slightly below zero
    for negative inputs. Used in modern architectures.
    """
    return scalar.multiply(x, sigmoid(x))


def silu_derivative(x: float) -> float:
    """Derivative of SiLU: sigmoid(x) + x × sigmoid(x) × (1 - sigmoid(x))."""
    s = sigmoid(x)
    return scalar.add(s, scalar.multiply(x, scalar.multiply(s, scalar.subtract(1.0, s))))


def elu(x: float, alpha: float = 1.0) -> float:
    """ELU: x if x > 0, else alpha × (exp(x) - 1).

    Like ReLU but with a smooth curve for negative inputs
    instead of a hard cutoff at zero. Used in DreamerV3.
    """
    if x > 0:
        return x
    return scalar.multiply(alpha, scalar.subtract(scalar.exp(x), 1.0))


def elu_derivative(x: float, alpha: float = 1.0) -> float:
    """Derivative of ELU: 1 if x > 0, else alpha × exp(x)."""
    if x > 0:
        return 1.0
    return scalar.multiply(alpha, scalar.exp(x))


def softplus(x: float) -> float:
    """log(1 + exp(x)) — a smooth approximation of ReLU.

    Always positive, approaches x for large x. Clamped to
    avoid computing exp of large numbers.
    """
    if x > 20.0:
        return x
    return scalar.log(scalar.add(1.0, scalar.exp(x)))


def softplus_derivative(x: float) -> float:
    """Derivative of softplus: sigmoid(x)."""
    return sigmoid(x)


def identity(x: float) -> float:
    """Pass-through — no transformation. Used for output layers."""
    return x


def identity_derivative(x: float) -> float:
    """Derivative of identity: always 1."""
    return 1.0


def layer_norm(v: Vector) -> Vector:
    """Normalize a vector so its average is 0 and its spread is 1.

    Prevents any one neuron from dominating by keeping all values
    in a similar range. Centers the values (subtract mean), then
    scales them (divide by standard deviation).
    """
    n = len(v)
    mean = vector.sum_all(v) / n
    centered = [scalar.subtract(x, mean) for x in v]
    variance = vector.sum_all([scalar.multiply(val, val) for val in centered]) / n
    eps = 1e-5  # prevents division by zero when variance is tiny
    std_inv = scalar.inverse(scalar.power(scalar.add(variance, eps), 0.5))
    return [scalar.multiply(val, std_inv) for val in centered]


def layer_norm_backward(grad_out: Vector, normed_input: Vector, original_input: Vector) -> Vector:
    """Backward pass for layer norm.

    Computes how the gradient flows back through the normalization.
    Each step of the forward pass (center, compute variance, scale)
    is reversed, accumulating how the gradient flows through each.
    The d_ prefix means "gradient with respect to."
    """
    n = len(grad_out)
    mean = vector.sum_all(original_input) / n
    centered = [scalar.subtract(x, mean) for x in original_input]
    variance = vector.sum_all([scalar.multiply(val, val) for val in centered]) / n
    eps = 1e-5
    std_inv = scalar.inverse(scalar.power(scalar.add(variance, eps), 0.5))

    # Gradient through the scaling step
    d_centered = [scalar.multiply(g, std_inv) for g in grad_out]

    # Gradient through the variance computation
    d_variance = 0.0
    var_factor = scalar.multiply(-0.5, scalar.power(scalar.add(variance, eps), -1.5))
    for dim in range(n):
        d_variance = scalar.add(d_variance,
                                scalar.multiply(grad_out[dim],
                                                scalar.multiply(centered[dim], var_factor)))

    # Gradient through the mean subtraction
    d_mean = 0.0
    for dim in range(n):
        d_mean = scalar.subtract(d_mean, d_centered[dim])
    sum_centered = vector.sum_all(centered)
    d_mean = scalar.add(d_mean,
                        scalar.multiply(d_variance, scalar.multiply(-2.0 / n, sum_centered)))

    # Combine all gradient paths into the final input gradient
    d_input = []
    for dim in range(n):
        grad = scalar.add(d_centered[dim],
                          scalar.add(scalar.multiply(d_variance,
                                                     scalar.multiply(2.0 / n, centered[dim])),
                                     d_mean / n))
        d_input.append(grad)
    return d_input


def softmax(v: Vector) -> Vector:
    """Convert a vector of scores into probabilities that sum to 1.

    Higher scores get higher probabilities, but even low scores
    get a small chance. Subtracting the max first prevents overflow
    without changing the result (the math cancels out).
    """
    max_score = vector.max_val(v)
    exps = [scalar.exp(scalar.subtract(x, max_score)) for x in v]
    total = vector.sum_all(exps)
    return [scalar.multiply(e, scalar.inverse(total)) for e in exps]
