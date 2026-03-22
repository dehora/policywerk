"""Tests for primitive operations."""

import io

from policywerk.primitives import scalar, vector, matrix, activations, losses, random
from policywerk.primitives.progress import Spinner


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


class TestActivations:
    def test_sigmoid(self):
        assert abs(activations.sigmoid(0.0) - 0.5) < 1e-10
        assert activations.sigmoid(100.0) > 0.99
        assert activations.sigmoid(-100.0) < 0.01

    def test_relu(self):
        assert activations.relu(5.0) == 5.0
        assert activations.relu(-5.0) == 0.0
        assert activations.relu(0.0) == 0.0

    def test_elu(self):
        assert activations.elu(5.0) == 5.0
        assert activations.elu(0.0) < 0.001
        assert activations.elu(-1.0) < 0.0

    def test_softmax(self):
        probs = activations.softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[2] > probs[1] > probs[0]

    def test_tanh(self):
        assert abs(activations.tanh_(0.0)) < 1e-10
        assert activations.tanh_(100.0) > 0.99
        assert activations.tanh_(-100.0) < -0.99


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
