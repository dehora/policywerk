"""Tests for building blocks."""

from policywerk.primitives import activations
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.value_functions import TabularV, TabularQ
from policywerk.building_blocks.policies import greedy, epsilon_greedy, softmax_policy
from policywerk.building_blocks.traces import EligibilityTrace
from policywerk.building_blocks.returns import discount_return, n_step_return, gae
from policywerk.building_blocks.replay_buffer import ReplayBuffer
from policywerk.building_blocks.distributions import Categorical, Gaussian
from policywerk.building_blocks.mdp import State, Transition
from policywerk.building_blocks.dense import create_dense, dense_forward
from policywerk.building_blocks.network import create_network, network_forward
from policywerk.building_blocks.grad import backward, numerical_gradient_check
from policywerk.building_blocks.optimizers import sgd_update, adam_update, create_adam_state
from policywerk.primitives.losses import mse


class TestTabularV:
    def test_get_set(self):
        v = TabularV()
        assert v.get("s1") == 0.0
        v.set("s1", 5.0)
        assert v.get("s1") == 5.0

    def test_update(self):
        v = TabularV()
        v.update("s1", 3.0)
        v.update("s1", 2.0)
        assert v.get("s1") == 5.0

    def test_max_change(self):
        v1 = TabularV()
        v2 = TabularV()
        v1.set("s1", 1.0)
        v2.set("s1", 4.0)
        assert v1.max_change(v2) == 3.0


class TestTabularQ:
    def test_get_set(self):
        q = TabularQ()
        assert q.get("s1", 0) == 0.0
        q.set("s1", 0, 10.0)
        assert q.get("s1", 0) == 10.0

    def test_best_action(self):
        q = TabularQ()
        q.set("s1", 0, 1.0)
        q.set("s1", 1, 5.0)
        q.set("s1", 2, 3.0)
        assert q.best_action("s1", 3) == 1

    def test_max_value(self):
        q = TabularQ()
        q.set("s1", 0, 1.0)
        q.set("s1", 1, 5.0)
        assert q.max_value("s1", 2) == 5.0


class TestPolicies:
    def test_greedy(self):
        assert greedy([1.0, 3.0, 2.0]) == 1

    def test_epsilon_greedy_exploit(self):
        rng = create_rng(42)
        # With epsilon=0, should always be greedy
        actions = [epsilon_greedy(rng, [1.0, 5.0, 2.0], 0.0) for _ in range(10)]
        assert all(a == 1 for a in actions)

    def test_softmax_policy(self):
        rng = create_rng(42)
        # With very low temperature, should be near-greedy
        actions = [softmax_policy(rng, [1.0, 100.0, 1.0], temperature=0.01) for _ in range(20)]
        assert sum(1 for a in actions if a == 1) > 15


class TestEligibilityTrace:
    def test_visit_and_decay(self):
        trace = EligibilityTrace(gamma=0.9, lam=0.8)
        trace.visit("s1")
        assert trace.get("s1") == 1.0
        trace.decay()
        assert abs(trace.get("s1") - 0.72) < 1e-10  # 0.9 * 0.8 = 0.72

    def test_reset(self):
        trace = EligibilityTrace(gamma=0.9, lam=0.8)
        trace.visit("s1")
        trace.reset()
        assert trace.get("s1") == 0.0


class TestReturns:
    def test_discount_return(self):
        # G = 1 + 0.9*2 + 0.81*3 = 1 + 1.8 + 2.43 = 5.23
        g = discount_return([1.0, 2.0, 3.0], gamma=0.9)
        assert abs(g - 5.23) < 1e-10

    def test_n_step_return(self):
        # G = 1 + 0.9*2 + 0.81*V = 1 + 1.8 + 0.81*10 = 10.9
        g = n_step_return([1.0, 2.0], bootstrap_value=10.0, gamma=0.9)
        assert abs(g - 10.9) < 1e-10

    def test_gae(self):
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        advantages = gae(rewards, values, next_value=0.5, gamma=0.99, lam=0.95)
        assert len(advantages) == 3
        # All advantages should be positive (rewards > 0, values low)
        assert all(a > 0 for a in advantages)


class TestReplayBuffer:
    def test_add_and_sample(self):
        buf = ReplayBuffer(capacity=10)
        s = State(features=[0.0], label="s")
        for i in range(5):
            buf.add(Transition(s, 0, float(i), s, False))
        assert len(buf) == 5
        rng = create_rng(42)
        batch = buf.sample(rng, 3)
        assert len(batch) == 3

    def test_circular(self):
        buf = ReplayBuffer(capacity=3)
        s = State(features=[0.0], label="s")
        for i in range(5):
            buf.add(Transition(s, 0, float(i), s, False))
        assert len(buf) == 3


class TestDistributions:
    def test_categorical(self):
        cat = Categorical(logits=[0.0, 0.0, 100.0])
        assert cat.probs[2] > 0.99
        rng = create_rng(42)
        samples = [cat.sample(rng) for _ in range(20)]
        assert all(s == 2 for s in samples)

    def test_categorical_entropy(self):
        # Uniform distribution should have maximum entropy
        uniform = Categorical(logits=[0.0, 0.0, 0.0])
        peaked = Categorical(logits=[0.0, 0.0, 10.0])
        assert uniform.entropy() > peaked.entropy()

    def test_gaussian(self):
        g = Gaussian(mean=[0.0], std=[1.0])
        rng = create_rng(42)
        samples = [g.sample(rng)[0] for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.2

    def test_gaussian_log_prob(self):
        g = Gaussian(mean=[0.0], std=[1.0])
        # Log prob at mean should be higher than far away
        assert g.log_prob([0.0]) > g.log_prob([5.0])


class TestNeuralNetwork:
    def test_dense_forward(self):
        rng = create_rng(42)
        layer = create_dense(rng, 3, 2)
        output, cache = dense_forward(layer, [1.0, 2.0, 3.0], activations.relu)
        assert len(output) == 2
        assert len(cache.z) == 2

    def test_network_forward(self):
        rng = create_rng(42)
        net = create_network(rng, [4, 8, 2], [activations.relu, activations.identity])
        output, cache = network_forward(net, [1.0, 2.0, 3.0, 4.0])
        assert len(output) == 2
        assert len(cache.layer_caches) == 2

    def test_gradient_check(self):
        rng = create_rng(42)
        net = create_network(rng, [3, 4, 2], [activations.sigmoid, activations.identity])
        max_error = numerical_gradient_check(
            net, [1.0, 0.5, -0.5], [1.0, 0.0], mse
        )
        assert max_error < 1e-4

    def test_sgd_reduces_loss(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 4, 1], [activations.relu, activations.identity])
        inputs = [1.0, 2.0]
        targets = [5.0]

        output1, cache1 = network_forward(net, inputs)
        loss1 = mse(output1, targets)

        loss_grad = [2.0 * (output1[i] - targets[i]) / len(targets) for i in range(len(targets))]
        grads = backward(net, cache1, loss_grad)
        sgd_update(net, grads, learning_rate=0.01)

        output2, _ = network_forward(net, inputs)
        loss2 = mse(output2, targets)
        assert loss2 < loss1

    def test_adam_reduces_loss(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 4, 1], [activations.relu, activations.identity])
        adam_states = create_adam_state(net)
        inputs = [1.0, 2.0]
        targets = [5.0]

        output1, _ = network_forward(net, inputs)
        loss1 = mse(output1, targets)

        for _ in range(10):
            output, cache = network_forward(net, inputs)
            loss_grad = [2.0 * (output[i] - targets[i]) / len(targets) for i in range(len(targets))]
            grads = backward(net, cache, loss_grad)
            adam_update(net, grads, adam_states, learning_rate=0.01)

        output_final, _ = network_forward(net, inputs)
        loss_final = mse(output_final, targets)
        assert loss_final < loss1
