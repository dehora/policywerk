"""Level 0: Vector operations.

Operations on lists of floats — dot products, element-wise ops,
scaling. Built from scalar operations.

A vector is just a list of numbers. These operations let us work
with entire lists at once rather than one number at a time.
"""

from policywerk.primitives import scalar

Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    """Multiply matching elements and sum the results.

    Measures how similar two vectors are, or how much one
    points in the direction of the other. This is the most
    important operation in neural networks — every neuron
    computes a dot product of its weights and inputs.

    Example: dot([1, 2, 3], [4, 5, 6]) = 1*4 + 2*5 + 3*6 = 32
    """
    result = 0.0
    for i in range(len(a)):
        result = scalar.add(result, scalar.multiply(a[i], b[i]))
    return result


def add(a: Vector, b: Vector) -> Vector:
    """Add two vectors element by element."""
    return [scalar.add(a[i], b[i]) for i in range(len(a))]


def subtract(a: Vector, b: Vector) -> Vector:
    """Subtract b from a, element by element."""
    return [scalar.subtract(a[i], b[i]) for i in range(len(a))]


def scale(c: float, v: Vector) -> Vector:
    """Multiply every element by the same number."""
    return [scalar.multiply(c, x) for x in v]


def elementwise(fn, a: Vector, b: Vector) -> Vector:
    """Apply a function to matching pairs from two vectors.

    Unlike dot (which sums the results into one number), this
    returns a vector of the same length.
    """
    return [fn(a[i], b[i]) for i in range(len(a))]


def apply(fn, v: Vector) -> Vector:
    """Run a function on each element of a vector."""
    return [fn(x) for x in v]


def magnitude(v: Vector) -> float:
    """Length of a vector — the Euclidean norm (Pythagorean distance from origin)."""
    return scalar.power(dot(v, v), 0.5)


def zeros(n: int) -> Vector:
    """Create a vector of n zeros."""
    return [0.0] * n


def ones(n: int) -> Vector:
    """Create a vector of n ones."""
    return [1.0] * n


def sum_all(v: Vector) -> float:
    """Add up all elements in a vector."""
    result = 0.0
    for x in v:
        result = scalar.add(result, x)
    return result


def max_val(v: Vector) -> float:
    """Return the largest element."""
    result = v[0]
    for x in v[1:]:
        if x > result:
            result = x
    return result


def argmax(v: Vector) -> int:
    """Return the index of the largest element.

    Used to pick the best action: if Q-values are [0.1, 0.9, 0.3],
    argmax returns 1 (the index of 0.9).
    """
    best_idx = 0
    best_val = v[0]
    for i in range(1, len(v)):
        if v[i] > best_val:
            best_val = v[i]
            best_idx = i
    return best_idx


def concat(a: Vector, b: Vector) -> Vector:
    """Join two vectors end to end into one longer vector."""
    return a + b


def slice_vec(v: Vector, start: int, end: int) -> Vector:
    """Return elements v[start:end]."""
    return v[start:end]
