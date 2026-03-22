"""Level 0: Loss functions.

Measure how far predictions are from targets — MSE, cross-entropy,
Huber loss, symlog. Each includes its derivative for backprop.
Built from scalar and vector operations.
"""

import math

from policywerk.primitives import scalar, vector

Vector = list[float]


def mse(predicted: Vector, actual: Vector) -> float:
    """Mean squared error between predicted and actual vectors."""
    diffs = vector.subtract(predicted, actual)
    squared = [scalar.multiply(d, d) for d in diffs]
    return scalar.multiply(vector.sum_all(squared), scalar.inverse(len(squared)))


def mse_derivative(predicted: Vector, actual: Vector) -> Vector:
    """Gradient of MSE with respect to each predicted value."""
    n = len(predicted)
    factor = 2.0 / n
    return [scalar.multiply(factor, scalar.subtract(p, a))
            for p, a in zip(predicted, actual)]


def cross_entropy(predicted: Vector, actual: Vector) -> float:
    """Cross-entropy loss — measures divergence between predicted and actual distributions."""
    terms = [scalar.multiply(a, scalar.log(p))
             for p, a in zip(predicted, actual)]
    return scalar.negate(vector.sum_all(terms))


def cross_entropy_derivative(predicted: Vector, actual: Vector) -> Vector:
    """Gradient of cross-entropy with respect to each predicted value."""
    return [scalar.negate(scalar.multiply(a, scalar.inverse(p)))
            for p, a in zip(predicted, actual)]


def huber(predicted: Vector, actual: Vector, delta: float = 1.0) -> float:
    """Huber loss — quadratic for small errors, linear for large ones.

    Used by DQN to prevent gradient explosion from large TD errors.
    """
    total = 0.0
    for p, a in zip(predicted, actual):
        diff = scalar.subtract(p, a)
        abs_diff = scalar.abs_val(diff)
        if abs_diff <= delta:
            total = scalar.add(total, scalar.multiply(0.5, scalar.multiply(diff, diff)))
        else:
            total = scalar.add(total,
                               scalar.subtract(scalar.multiply(delta, abs_diff),
                                               scalar.multiply(0.5, scalar.multiply(delta, delta))))
    return scalar.multiply(total, scalar.inverse(len(predicted)))


def huber_derivative(predicted: Vector, actual: Vector, delta: float = 1.0) -> Vector:
    """Gradient of Huber loss with respect to each predicted value."""
    n = len(predicted)
    grads = []
    for p, a in zip(predicted, actual):
        diff = scalar.subtract(p, a)
        abs_diff = scalar.abs_val(diff)
        if abs_diff <= delta:
            grads.append(scalar.multiply(diff, scalar.inverse(n)))
        else:
            grads.append(scalar.multiply(scalar.multiply(delta, scalar.sign(diff)),
                                         scalar.inverse(n)))
    return grads


def symlog(x: float) -> float:
    """Symmetric logarithm: sign(x) * log(|x| + 1).

    DreamerV3 uses symlog to compress targets into a learnable range
    while preserving sign. Large rewards/values get compressed,
    small ones stay nearly linear.
    """
    return scalar.multiply(scalar.sign(x), scalar.log(scalar.add(scalar.abs_val(x), 1.0)))


def symexp(x: float) -> float:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return scalar.multiply(scalar.sign(x), scalar.subtract(scalar.exp(scalar.abs_val(x)), 1.0))


def twohot_encode(x: float, num_bins: int, lo: float, hi: float) -> Vector:
    """Encode a scalar as a two-hot distribution over bins.

    DreamerV3 represents continuous values as distributions over
    discrete bins. The value is placed between the two nearest bins
    with weights that interpolate linearly.
    """
    bin_width = scalar.multiply(scalar.subtract(hi, lo), scalar.inverse(num_bins - 1))
    # Which bin does x fall into?
    pos = scalar.multiply(scalar.subtract(scalar.clamp(x, lo, hi), lo), scalar.inverse(bin_width))
    lower = int(pos)
    if lower >= num_bins - 1:
        lower = num_bins - 2
    upper = lower + 1
    weight_upper = scalar.subtract(pos, float(lower))
    weight_lower = scalar.subtract(1.0, weight_upper)
    result = vector.zeros(num_bins)
    result[lower] = weight_lower
    result[upper] = weight_upper
    return result


def twohot_decode(probs: Vector, lo: float, hi: float) -> float:
    """Decode a distribution over bins back to a scalar value.

    The expected value under the distribution gives the decoded scalar.
    """
    num_bins = len(probs)
    bin_width = scalar.multiply(scalar.subtract(hi, lo), scalar.inverse(num_bins - 1))
    result = 0.0
    for i in range(num_bins):
        bin_center = scalar.add(lo, scalar.multiply(float(i), bin_width))
        result = scalar.add(result, scalar.multiply(probs[i], bin_center))
    return result
