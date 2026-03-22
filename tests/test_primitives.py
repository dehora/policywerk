"""Tests for primitive operations."""

import io

from policywerk.primitives import scalar, vector, matrix, activations, losses, random
from policywerk.primitives.progress import Spinner, progress_bar


class TestScalar:
    def test_basic_arithmetic(self):
        assert scalar.add(2.0, 3.0) == 5.0
        assert scalar.subtract(5.0, 3.0) == 2.0
        assert scalar.multiply(2.0, 3.0) == 6.0
        assert scalar.negate(3.0) == -3.0
        assert scalar.inverse(2.0) == 0.5

    def test_exp_log(self):
        assert abs(scalar.exp(0.0) - 1.0) < 1e-10
        assert abs(scalar.log(1.0) - 0.0) < 1e-10
        assert abs(scalar.exp(scalar.log(5.0)) - 5.0) < 1e-10

    def test_clamp(self):
        assert scalar.clamp(5.0, 0.0, 10.0) == 5.0
        assert scalar.clamp(-1.0, 0.0, 10.0) == 0.0
        assert scalar.clamp(11.0, 0.0, 10.0) == 10.0

    def test_abs_val(self):
        assert scalar.abs_val(3.0) == 3.0
        assert scalar.abs_val(-3.0) == 3.0
        assert scalar.abs_val(0.0) == 0.0

    def test_sign(self):
        assert scalar.sign(5.0) == 1.0
        assert scalar.sign(-5.0) == -1.0
        assert scalar.sign(0.0) == 0.0


class TestVector:
    def test_dot(self):
        assert vector.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0

    def test_add_subtract(self):
        assert vector.add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
        assert vector.subtract([5.0, 3.0], [2.0, 1.0]) == [3.0, 2.0]

    def test_scale(self):
        assert vector.scale(2.0, [1.0, 2.0, 3.0]) == [2.0, 4.0, 6.0]

    def test_argmax(self):
        assert vector.argmax([1.0, 3.0, 2.0]) == 1
        assert vector.argmax([5.0, 1.0, 2.0]) == 0

    def test_concat(self):
        assert vector.concat([1.0, 2.0], [3.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]

    def test_slice_vec(self):
        assert vector.slice_vec([1.0, 2.0, 3.0, 4.0], 1, 3) == [2.0, 3.0]

    def test_sum_all(self):
        assert vector.sum_all([1.0, 2.0, 3.0]) == 6.0

    def test_zeros_ones(self):
        assert vector.zeros(3) == [0.0, 0.0, 0.0]
        assert vector.ones(3) == [1.0, 1.0, 1.0]


class TestMatrix:
    def test_mat_vec(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        v = [1.0, 1.0]
        assert matrix.mat_vec(M, v) == [3.0, 7.0]

    def test_transpose(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        T = matrix.transpose(M)
        assert T == [[1.0, 3.0], [2.0, 4.0]]

    def test_outer(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        assert matrix.outer(a, b) == [[3.0, 4.0], [6.0, 8.0]]

    def test_flatten_reshape(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        flat = matrix.flatten(M)
        assert flat == [1.0, 2.0, 3.0, 4.0]
        assert matrix.reshape(flat, 2, 2) == M

    def test_mat_mat(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = matrix.mat_mat(A, B)
        assert result == [[19.0, 22.0], [43.0, 50.0]]

    def test_tensor3d_zeros(self):
        t = matrix.tensor3d_zeros(2, 3, 4)
        assert len(t) == 2
        assert len(t[0]) == 3
        assert len(t[0][0]) == 4
        assert t[0][0][0] == 0.0

    def test_tensor3d_flatten_and_reshape(self):
        t = matrix.tensor3d_zeros(2, 3, 4)
        t[0][1][2] = 5.0
        t[1][2][3] = 7.0
        flat = matrix.tensor3d_flatten(t)
        assert len(flat) == 2 * 3 * 4
        restored = matrix.tensor3d_reshape(flat, 2, 3, 4)
        assert restored[0][1][2] == 5.0
        assert restored[1][2][3] == 7.0


class TestActivations:
    def test_sigmoid(self):
        assert abs(activations.sigmoid(0.0) - 0.5) < 1e-10
        assert activations.sigmoid(100.0) > 0.99
        assert activations.sigmoid(-100.0) < 0.01

    def test_sigmoid_derivative(self):
        assert abs(activations.sigmoid_derivative(0.0) - 0.25) < 1e-10

    def test_relu(self):
        assert activations.relu(5.0) == 5.0
        assert activations.relu(-5.0) == 0.0
        assert activations.relu(0.0) == 0.0

    def test_relu_derivative(self):
        assert activations.relu_derivative(5.0) == 1.0
        assert activations.relu_derivative(-5.0) == 0.0

    def test_elu(self):
        assert activations.elu(5.0) == 5.0
        assert activations.elu(0.0) < 0.001
        assert activations.elu(-1.0) < 0.0

    def test_elu_derivative(self):
        assert activations.elu_derivative(5.0) == 1.0
        import math
        assert abs(activations.elu_derivative(-1.0, alpha=1.0) - math.exp(-1.0)) < 1e-10

    def test_softmax(self):
        probs = activations.softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[2] > probs[1] > probs[0]

    def test_tanh(self):
        assert abs(activations.tanh_(0.0)) < 1e-10
        assert activations.tanh_(100.0) > 0.99
        assert activations.tanh_(-100.0) < -0.99

    def test_tanh_derivative(self):
        assert abs(activations.tanh_derivative(0.0) - 1.0) < 1e-10

    def test_silu(self):
        assert abs(activations.silu(0.0)) < 1e-10

    def test_silu_derivative(self):
        assert abs(activations.silu_derivative(0.0) - 0.5) < 1e-10

    def test_softplus(self):
        import math
        assert abs(activations.softplus(0.0) - math.log(2.0)) < 1e-10

    def test_identity_and_derivative(self):
        assert activations.identity(3.14) == 3.14
        assert activations.identity_derivative(3.14) == 1.0

    def test_layer_norm(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = activations.layer_norm(v)
        mean = sum(normed) / len(normed)
        variance = sum((x - mean) ** 2 for x in normed) / len(normed)
        assert abs(mean) < 1e-5
        assert abs(variance - 1.0) < 1e-3

    def test_layer_norm_backward(self):
        """Verify layer_norm gradients via finite differences."""
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = activations.layer_norm(v)
        grad_out = [1.0, 0.0, -1.0, 0.5, -0.5]
        d_input = activations.layer_norm_backward(grad_out, normed, v)
        assert len(d_input) == len(v)

        # Finite-difference check for each input dimension
        eps = 1e-5
        for i in range(len(v)):
            v_plus = list(v)
            v_plus[i] += eps
            v_minus = list(v)
            v_minus[i] -= eps
            normed_plus = activations.layer_norm(v_plus)
            normed_minus = activations.layer_norm(v_minus)
            # Loss = sum(grad_out * normed)
            loss_plus = sum(g * n for g, n in zip(grad_out, normed_plus))
            loss_minus = sum(g * n for g, n in zip(grad_out, normed_minus))
            numerical = (loss_plus - loss_minus) / (2 * eps)
            assert abs(numerical - d_input[i]) < 1e-4, f"d_input[{i}]: {numerical} vs {d_input[i]}"

    def test_step(self):
        assert activations.step(1.0) == 1.0
        assert activations.step(-1.0) == 0.0
        assert activations.step(0.0) == 1.0


class TestLosses:
    def test_mse(self):
        assert losses.mse([1.0, 2.0], [1.0, 2.0]) == 0.0
        assert losses.mse([1.0, 2.0], [3.0, 4.0]) == 4.0

    def test_huber(self):
        # Small errors — should behave like 0.5 * MSE
        h = losses.huber([1.0], [1.5], delta=1.0)
        assert abs(h - 0.125) < 1e-10
        # Large errors — should be linear
        h = losses.huber([1.0], [10.0], delta=1.0)
        assert h < losses.mse([1.0], [10.0])

    def test_huber_derivative_small(self):
        # Within delta: gradient = diff / n
        grads = losses.huber_derivative([1.5], [1.0], delta=1.0)
        assert abs(grads[0] - 0.5) < 1e-10  # (1.5 - 1.0) / 1

    def test_huber_derivative_large(self):
        # Beyond delta: gradient is clipped to delta * sign(diff) / n
        grads = losses.huber_derivative([5.0], [1.0], delta=1.0)
        assert abs(grads[0] - 1.0) < 1e-10  # delta * sign(4.0) / 1

    def test_cross_entropy_derivative(self):
        """Verify cross-entropy gradient: d/dp[-a*log(p)] = -a/p."""
        predicted = [0.7, 0.2, 0.1]
        actual = [1.0, 0.0, 0.0]
        grads = losses.cross_entropy_derivative(predicted, actual)
        assert len(grads) == 3
        # For actual[0]=1.0, predicted[0]=0.7: gradient = -1.0/0.7
        assert abs(grads[0] - (-1.0 / 0.7)) < 1e-10
        # For actual[1]=0.0: gradient = 0 (no contribution)
        assert abs(grads[1] - 0.0) < 1e-10
        assert abs(grads[2] - 0.0) < 1e-10

    def test_symlog_symexp_inverse(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            assert abs(losses.symexp(losses.symlog(x)) - x) < 1e-10

    def test_twohot_roundtrip(self):
        for x in [-5.0, 0.0, 3.7, 10.0]:
            encoded = losses.twohot_encode(x, num_bins=21, lo=-10.0, hi=10.0)
            decoded = losses.twohot_decode(encoded, lo=-10.0, hi=10.0)
            assert abs(decoded - x) < 0.01


class TestRandom:
    def test_deterministic(self):
        rng1 = random.create_rng(42)
        rng2 = random.create_rng(42)
        assert random.uniform(rng1, 0.0, 1.0) == random.uniform(rng2, 0.0, 1.0)

    def test_normal(self):
        rng = random.create_rng(42)
        vals = [random.normal(rng) for _ in range(1000)]
        mean = sum(vals) / len(vals)
        assert abs(mean) < 0.2  # should be near 0

    def test_sample_categorical(self):
        rng = random.create_rng(42)
        probs = [0.0, 0.0, 1.0]
        for _ in range(10):
            assert random.sample_categorical(rng, probs) == 2

    def test_choice(self):
        rng = random.create_rng(42)
        val = random.choice(rng, 5)
        assert 0 <= val < 5

    def test_normal_vector(self):
        rng = random.create_rng(42)
        v = random.normal_vector(rng, 10)
        assert len(v) == 10


class TestProgress:
    def test_progress_bar(self):
        buf = io.StringIO()
        progress_bar(epoch=5, total=10, loss=0.1234, stream=buf)
        output = buf.getvalue()
        assert "5/10" in output
        assert "0.1234" in output


class TestSpinner:
    def test_spinner_done_message(self):
        buf = io.StringIO()
        with Spinner("Working", stream=buf):
            pass
        output = buf.getvalue()
        # Final line should contain "done." without TTY padding
        last_line = output.strip().split("\n")[-1]
        assert "done." in last_line
        # StringIO is not a TTY, so no trailing padding spaces
        assert last_line == last_line.rstrip()

    def test_spinner_error_message(self):
        buf = io.StringIO()
        try:
            with Spinner("Working", stream=buf):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        output = buf.getvalue()
        last_line = output.strip().split("\n")[-1]
        assert "failed" in last_line
        assert "done." not in last_line

    def test_spinner_no_cr_on_non_tty(self):
        """Non-TTY streams should have no carriage returns or spinner frames."""
        import time
        buf = io.StringIO()
        with Spinner("Working", stream=buf):
            time.sleep(0.3)  # give spinner thread time to run (if it were to)
        output = buf.getvalue()
        assert "\r" not in output, "Non-TTY output should contain no carriage returns"
        assert "⠋" not in output, "Non-TTY output should contain no spinner characters"
        assert "|" not in output or "done." in output, "No ASCII spinner frames in output"
        # Should be exactly one clean line
        lines = [l for l in output.split("\n") if l.strip()]
        assert len(lines) == 1
        assert "done." in lines[0]
