"""Level 0: Vector operations.

Operations on lists of floats — dot products, element-wise ops,
scaling. Built from scalar operations.
"""

from policywerk.primitives import scalar

Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    result = 0.0
    for i in range(len(a)):
        result = scalar.add(result, scalar.multiply(a[i], b[i]))
    return result


def add(a: Vector, b: Vector) -> Vector:
    return [scalar.add(a[i], b[i]) for i in range(len(a))]


def subtract(a: Vector, b: Vector) -> Vector:
    return [scalar.subtract(a[i], b[i]) for i in range(len(a))]


def scale(c: float, v: Vector) -> Vector:
    return [scalar.multiply(c, x) for x in v]


def elementwise(fn, a: Vector, b: Vector) -> Vector:
    return [fn(a[i], b[i]) for i in range(len(a))]


def apply(fn, v: Vector) -> Vector:
    return [fn(x) for x in v]


def magnitude(v: Vector) -> float:
    """Length of a vector — the Euclidean norm."""
    return scalar.power(dot(v, v), 0.5)


def zeros(n: int) -> Vector:
    return [0.0] * n


def ones(n: int) -> Vector:
    return [1.0] * n


def sum_all(v: Vector) -> float:
    result = 0.0
    for x in v:
        result = scalar.add(result, x)
    return result


def max_val(v: Vector) -> float:
    result = v[0]
    for x in v[1:]:
        if x > result:
            result = x
    return result


def argmax(v: Vector) -> int:
    """Return the index of the largest element."""
    best_idx = 0
    best_val = v[0]
    for i in range(1, len(v)):
        if v[i] > best_val:
            best_val = v[i]
            best_idx = i
    return best_idx


def concat(a: Vector, b: Vector) -> Vector:
    """Concatenate two vectors into one."""
    return a + b


def slice_vec(v: Vector, start: int, end: int) -> Vector:
    """Return elements v[start:end]."""
    return v[start:end]
