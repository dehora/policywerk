"""Level 0: Matrix operations.

Operations on lists of lists of floats — matrix-vector multiply,
transpose, outer product. Built from vector operations.

A matrix is a grid of numbers (rows × columns). These operations
let neural network layers transform entire vectors of inputs
into vectors of outputs in one step.
"""

from policywerk.primitives import scalar, vector

Matrix = list[list[float]]
Vector = list[float]


def mat_vec(M: Matrix, v: Vector) -> Vector:
    """Matrix-vector multiply: each row of M is dotted with v.

    This is how a neural network layer transforms its input —
    each output value is the dot product of one row of weights
    with the input vector.
    """
    return [vector.dot(row, v) for row in M]


def mat_mat(A: Matrix, B: Matrix) -> Matrix:
    """Matrix-matrix multiply: combine two transformations into one."""
    B_T = transpose(B)
    return [[vector.dot(a_row, b_col) for b_col in B_T] for a_row in A]


def transpose(M: Matrix) -> Matrix:
    """Flip rows and columns — row 0 becomes column 0, etc."""
    if not M:
        return []
    rows, cols = len(M), len(M[0])
    return [[M[row][col] for row in range(rows)] for col in range(cols)]


def outer(a: Vector, b: Vector) -> Matrix:
    """Outer product: make a matrix from two vectors.

    Every combination of a[i] * b[j] fills one cell.
    Used in backpropagation to compute weight gradients:
    how much each weight contributed to the output.
    """
    return [[scalar.multiply(ai, bj) for bj in b] for ai in a]


def zeros(rows: int, cols: int) -> Matrix:
    """Create a rows × cols matrix filled with zeros."""
    return [[0.0] * cols for _ in range(rows)]


def add(A: Matrix, B: Matrix) -> Matrix:
    """Add two matrices element by element."""
    return [vector.add(A[row], B[row]) for row in range(len(A))]


def scale(c: float, M: Matrix) -> Matrix:
    """Multiply every element in the matrix by c."""
    return [vector.scale(c, row) for row in M]


def flatten(M: Matrix) -> Vector:
    """Unroll a 2D matrix into a 1D vector, row by row."""
    result: Vector = []
    for row in M:
        result.extend(row)
    return result


def reshape(v: Vector, rows: int, cols: int) -> Matrix:
    """Fold a 1D vector back into a 2D matrix."""
    return [v[row * cols:(row + 1) * cols] for row in range(rows)]


# --- 3D tensor operations (channels × height × width) ---
# Used by convolutional layers where the input is a multi-channel
# image (e.g. 4 stacked game frames, each 16×16 pixels).

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
