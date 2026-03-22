"""Level 0: Matrix operations.

Operations on lists of lists of floats — matrix-vector multiply,
transpose, outer product. Built from vector operations.
"""

from policywerk.primitives import scalar, vector

Matrix = list[list[float]]
Vector = list[float]


def mat_vec(M: Matrix, v: Vector) -> Vector:
    return [vector.dot(row, v) for row in M]


def mat_mat(A: Matrix, B: Matrix) -> Matrix:
    B_T = transpose(B)
    return [[vector.dot(a_row, b_col) for b_col in B_T] for a_row in A]


def transpose(M: Matrix) -> Matrix:
    if not M:
        return []
    rows, cols = len(M), len(M[0])
    return [[M[row][col] for row in range(rows)] for col in range(cols)]


def outer(a: Vector, b: Vector) -> Matrix:
    return [[scalar.multiply(ai, bj) for bj in b] for ai in a]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def add(A: Matrix, B: Matrix) -> Matrix:
    return [vector.add(A[row], B[row]) for row in range(len(A))]


def scale(c: float, M: Matrix) -> Matrix:
    return [vector.scale(c, row) for row in M]


def flatten(M: Matrix) -> Vector:
    result: Vector = []
    for row in M:
        result.extend(row)
    return result


def reshape(v: Vector, rows: int, cols: int) -> Matrix:
    return [v[row * cols:(row + 1) * cols] for row in range(rows)]


# --- 3D tensor operations (channels × height × width) ---

Tensor3D = list[list[list[float]]]


def tensor3d_zeros(channels: int, height: int, width: int) -> Tensor3D:
    """Create a zero-filled 3D tensor."""
    return [[[0.0] * width for _ in range(height)] for _ in range(channels)]


def tensor3d_flatten(tensor: Tensor3D) -> Vector:
    """Flatten a 3D tensor (C × H × W) into a 1D vector."""
    result: Vector = []
    for channel in tensor:
        for row in channel:
            result.extend(row)
    return result


def tensor3d_reshape(v: Vector, channels: int, height: int, width: int) -> Tensor3D:
    """Reshape a vector back into a 3D tensor (C × H × W)."""
    tensor: Tensor3D = []
    offset = 0
    for _ in range(channels):
        channel = []
        for _ in range(height):
            channel.append(v[offset:offset + width])
            offset += width
        tensor.append(channel)
    return tensor
