"""Level 0: Loss functions.

Measure how far predictions are from targets. A loss of 0 means
perfect predictions; larger values mean worse predictions. The
network's goal during training is to make the loss as small as
possible.

Each loss includes its derivative—how the loss changes when
each prediction moves slightly. This is what the network uses
to figure out which direction to adjust its weights.
"""

import math

from policywerk.primitives import scalar, vector

Vector = list[float]


def mse(predicted: Vector, actual: Vector) -> float:
    """Mean squared error—average of (predicted - actual)² for each element.

    Penalizes large errors heavily (because of squaring).
    The most common loss for regression tasks.
    """
    diffs = vector.subtract(predicted, actual)
    squared = [scalar.multiply(d, d) for d in diffs]
    return scalar.multiply(vector.sum_all(squared), scalar.inverse(len(squared)))


def mse_derivative(predicted: Vector, actual: Vector) -> Vector:
    """How much the MSE changes when each prediction moves slightly.

    Points in the direction that would increase the error—the
    optimizer moves weights in the opposite direction to reduce it.
    """
    n = len(predicted)
    factor = 2.0 / n
    return [scalar.multiply(factor, scalar.subtract(p, a))
            for p, a in zip(predicted, actual)]


def cross_entropy(predicted: Vector, actual: Vector) -> float:
    """Cross-entropy—measures how surprised the model is by the actual answer.

    Lower when predictions match reality. Used when the output
    represents probabilities (e.g. after softmax).
    Formula: -Σ actual × log(predicted)
    """
    terms = [scalar.multiply(a, scalar.log(p))
             for p, a in zip(predicted, actual)]
    return scalar.negate(vector.sum_all(terms))


def cross_entropy_derivative(predicted: Vector, actual: Vector) -> Vector:
    """How much cross-entropy changes when each prediction moves."""
    return [scalar.negate(scalar.multiply(a, scalar.inverse(p)))
            for p, a in zip(predicted, actual)]


def huber(predicted: Vector, actual: Vector, delta: float = 1.0) -> float:
    """Huber loss—quadratic for small errors, linear for large ones.

    Combines the best of MSE (accurate for small errors) and absolute
    error (doesn't explode for large errors). Used by DQN to prevent
    gradient explosion from large TD errors.
    """
    total = 0.0
    for p, a in zip(predicted, actual):
        diff = scalar.subtract(p, a)
        abs_diff = scalar.abs_val(diff)
        if abs_diff <= delta:
            # Small error: use quadratic (like MSE)
            total = scalar.add(total, scalar.multiply(0.5, scalar.multiply(diff, diff)))
        else:
            # Large error: use linear (like absolute error)
            total = scalar.add(total,
                               scalar.subtract(scalar.multiply(delta, abs_diff),
                                               scalar.multiply(0.5, scalar.multiply(delta, delta))))
    return scalar.multiply(total, scalar.inverse(len(predicted)))


def huber_derivative(predicted: Vector, actual: Vector, delta: float = 1.0) -> Vector:
    """How much Huber loss changes when each prediction moves."""
    n = len(predicted)
    grads = []
    for p, a in zip(predicted, actual):
        diff = scalar.subtract(p, a)
        abs_diff = scalar.abs_val(diff)
        if abs_diff <= delta:
            grads.append(scalar.multiply(diff, scalar.inverse(n)))
        else:
            # Clip the gradient to delta—this is why Huber prevents explosion
            grads.append(scalar.multiply(scalar.multiply(delta, scalar.sign(diff)),
                                         scalar.inverse(n)))
    return grads


def symlog(x: float) -> float:
    """Symmetric logarithm: sign(x) × log(|x| + 1).

    Compresses large values while preserving sign. A reward of
    +1000 becomes ~6.9, while +1 stays ~0.7. DreamerV3 uses symlog
    to keep targets in a learnable range without losing information.
    """
    return scalar.multiply(scalar.sign(x), scalar.log(scalar.add(scalar.abs_val(x), 1.0)))


def symexp(x: float) -> float:
    """Inverse of symlog: sign(x) × (exp(|x|) - 1).

    Undoes the compression: symexp(symlog(x)) = x.
    """
    return scalar.multiply(scalar.sign(x), scalar.subtract(scalar.exp(scalar.abs_val(x)), 1.0))


def twohot_encode(x: float, num_bins: int, lo: float, hi: float) -> Vector:
    """Encode a scalar as a soft distribution over discrete bins.

    DreamerV3 represents continuous values (like predicted rewards)
    as distributions over fixed bins. The value is placed between
    the two nearest bins, with weights that interpolate linearly.

    Example: if bins are [0, 1, 2, 3] and x=1.7, the result is
    [0, 0.3, 0.7, 0]—weight split between bins 1 and 2.
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
    """Decode a bin distribution back to a scalar value.

    Takes the weighted average of bin centers—the expected value
    under the distribution.
    """
    num_bins = len(probs)
    bin_width = scalar.multiply(scalar.subtract(hi, lo), scalar.inverse(num_bins - 1))
    result = 0.0
    for i in range(num_bins):
        bin_center = scalar.add(lo, scalar.multiply(float(i), bin_width))
        result = scalar.add(result, scalar.multiply(probs[i], bin_center))
    return result
