"""Tests for building blocks."""

from policywerk.primitives import activations
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.value_functions import TabularV, TabularQ
from policywerk.building_blocks.policies import greedy, epsilon_greedy, softmax_policy, gaussian_policy
from policywerk.building_blocks.traces import EligibilityTrace
from policywerk.building_blocks.returns import discount_return, n_step_return, gae, lambda_return
from policywerk.building_blocks.replay_buffer import ReplayBuffer
from policywerk.building_blocks.distributions import Categorical, Gaussian
from policywerk.building_blocks.mdp import State, Transition
from policywerk.building_blocks.dense import create_dense, dense_forward
from policywerk.building_blocks.network import create_network, network_forward
from policywerk.building_blocks.grad import backward, numerical_gradient_check, LayerGradients
from policywerk.building_blocks.optimizers import sgd_update, sgd_momentum_update, adam_update, create_adam_state
from policywerk.building_blocks.neuron import create_neuron, forward as neuron_forward
from policywerk.building_blocks.conv import create_conv, conv_forward, conv_backward
from policywerk.building_blocks.pool import max_pool_forward, max_pool_backward, avg_pool_forward, avg_pool_backward
from policywerk.building_blocks.recurrent import create_gru, gru_forward, gru_backward
from policywerk.primitives.losses import mse
from policywerk.primitives import vector, matrix


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

    def test_softmax_policy_preferences_keyword(self):
        """Backward compatibility: preferences= keyword still works."""
        rng = create_rng(42)
        action = softmax_policy(rng, preferences=[1.0, 100.0, 1.0], temperature=0.01)
        assert action == 1


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

    def test_network_activation_fn_mismatch(self):
        """Creating a network with wrong number of activation functions raises."""
        rng = create_rng(42)
        import pytest
        # 2 layers need 2 activation functions, not 1
        with pytest.raises(ValueError, match="activation functions"):
            create_network(rng, [4, 8, 2], [activations.relu])

    def test_network_too_few_layer_sizes(self):
        """layer_sizes needs at least [input_dim, output_dim]."""
        rng = create_rng(42)
        import pytest
        with pytest.raises(ValueError, match="at least 2"):
            create_network(rng, [4], [activations.relu])
        with pytest.raises(ValueError, match="at least 2"):
            create_network(rng, [], [])

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


class TestNeuron:
    def test_create_neuron(self):
        rng = create_rng(42)
        neuron = create_neuron(rng, 5)
        assert len(neuron.weights) == 5
        assert neuron.bias == 0.0

    def test_neuron_forward(self):
        rng = create_rng(42)
        neuron = create_neuron(rng, 3)
        output = neuron_forward(neuron, [1.0, 2.0, 3.0], activations.relu)
        assert isinstance(output, float)


class TestConv:
    def test_create_conv(self):
        rng = create_rng(42)
        layer = create_conv(rng, in_channels=1, out_channels=2, kernel_size=3)
        assert len(layer.filters) == 2          # out_channels
        assert len(layer.filters[0]) == 1       # in_channels
        assert len(layer.filters[0][0]) == 3    # kernel height
        assert len(layer.filters[0][0][0]) == 3 # kernel width
        assert len(layer.biases) == 2

    def test_conv_forward(self):
        rng = create_rng(42)
        layer = create_conv(rng, in_channels=1, out_channels=2, kernel_size=3)
        # 1 channel, 4x4 input
        inputs = [[[float(r * 4 + c) for c in range(4)] for r in range(4)]]
        output, cache = conv_forward(layer, inputs, activations.relu)
        # output: 2 filters, (4-3+1)x(4-3+1) = 2x2
        assert len(output) == 2
        assert len(output[0]) == 2
        assert len(output[0][0]) == 2

    def test_conv_backward(self):
        rng = create_rng(42)
        layer = create_conv(rng, in_channels=1, out_channels=2, kernel_size=3)
        inputs = [[[float(r * 4 + c) for c in range(4)] for r in range(4)]]
        output, cache = conv_forward(layer, inputs, activations.relu)
        # output_grad same shape as output: 2 filters, 2x2
        output_grad = [[[1.0 for _ in range(2)] for _ in range(2)] for _ in range(2)]
        input_grad, filter_grads, bias_grads = conv_backward(
            layer, cache, output_grad, activations.relu_derivative,
        )
        # input_grad: 1 channel, 4x4
        assert len(input_grad) == 1
        assert len(input_grad[0]) == 4
        assert len(input_grad[0][0]) == 4
        # filter_grads: 2 filters, each 1 channel, 3x3
        assert len(filter_grads) == 2
        assert len(filter_grads[0]) == 1
        assert len(filter_grads[0][0]) == 3
        assert len(filter_grads[0][0][0]) == 3
        # bias_grads: 2
        assert len(bias_grads) == 2

    def test_conv_gradient_consistency(self):
        """Numerical gradient check on a tiny conv layer."""
        rng = create_rng(42)
        layer = create_conv(rng, in_channels=1, out_channels=1, kernel_size=2)
        inputs = [[[1.0, 2.0], [3.0, 4.0]]]
        eps = 1e-5

        # Forward to get analytical gradients
        output, cache = conv_forward(layer, inputs, activations.identity)
        output_grad = [[[1.0]]]
        _, filter_grads, bias_grads = conv_backward(
            layer, cache, output_grad, activations.identity_derivative,
        )

        # Numerical gradient for one filter weight
        orig = layer.filters[0][0][0][0]
        layer.filters[0][0][0][0] = orig + eps
        out_plus, _ = conv_forward(layer, inputs, activations.identity)
        layer.filters[0][0][0][0] = orig - eps
        out_minus, _ = conv_forward(layer, inputs, activations.identity)
        layer.filters[0][0][0][0] = orig
        numerical = (out_plus[0][0][0] - out_minus[0][0][0]) / (2 * eps)
        assert abs(numerical - filter_grads[0][0][0][0]) < 1e-4


class TestPool:
    def test_max_pool_forward(self):
        # 1 channel, 4x4 input
        inputs = [[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]]
        output, cache = max_pool_forward(inputs, pool_size=2, stride=2)
        assert len(output) == 1
        assert len(output[0]) == 2
        assert len(output[0][0]) == 2
        # Max values in each 2x2 block
        assert output[0][0][0] == 6.0
        assert output[0][0][1] == 8.0
        assert output[0][1][0] == 14.0
        assert output[0][1][1] == 16.0

    def test_max_pool_backward(self):
        inputs = [[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]]
        output, cache = max_pool_forward(inputs, pool_size=2, stride=2)
        output_grad = [[[1.0, 1.0], [1.0, 1.0]]]
        input_grad = max_pool_backward(output_grad, cache)
        assert len(input_grad) == 1
        assert len(input_grad[0]) == 4
        assert len(input_grad[0][0]) == 4
        # Gradient flows only to max elements (6,8,14,16)
        assert input_grad[0][1][1] == 1.0  # position of 6
        assert input_grad[0][1][3] == 1.0  # position of 8
        assert input_grad[0][3][1] == 1.0  # position of 14
        assert input_grad[0][3][3] == 1.0  # position of 16
        # Non-max elements get zero gradient
        assert input_grad[0][0][0] == 0.0
        assert input_grad[0][0][1] == 0.0

    def test_avg_pool_forward(self):
        inputs = [[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]]
        output, cache = avg_pool_forward(inputs, pool_size=2, stride=2)
        assert len(output) == 1
        assert len(output[0]) == 2
        assert len(output[0][0]) == 2
        # Average of each 2x2 block
        assert abs(output[0][0][0] - 3.5) < 1e-10   # (1+2+5+6)/4
        assert abs(output[0][0][1] - 5.5) < 1e-10   # (3+4+7+8)/4
        assert abs(output[0][1][0] - 11.5) < 1e-10  # (9+10+13+14)/4
        assert abs(output[0][1][1] - 13.5) < 1e-10  # (11+12+15+16)/4

    def test_avg_pool_backward(self):
        inputs = [[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]]
        _, cache = avg_pool_forward(inputs, pool_size=2, stride=2)
        output_grad = [[[4.0, 4.0], [4.0, 4.0]]]
        input_grad = avg_pool_backward(output_grad, cache, pool_size=2, stride=2)
        assert len(input_grad) == 1
        assert len(input_grad[0]) == 4
        # Each element in a 2x2 block gets gradient / 4
        assert abs(input_grad[0][0][0] - 1.0) < 1e-10
        assert abs(input_grad[0][0][1] - 1.0) < 1e-10
        assert abs(input_grad[0][1][0] - 1.0) < 1e-10
        assert abs(input_grad[0][1][1] - 1.0) < 1e-10


class TestRecurrent:
    def test_create_gru(self):
        rng = create_rng(42)
        layer = create_gru(rng, input_size=4, hidden_size=3)
        combined = 3 + 4  # hidden_size + input_size
        assert len(layer.W_z) == 3
        assert len(layer.W_z[0]) == combined
        assert len(layer.W_r) == 3
        assert len(layer.W_r[0]) == combined
        assert len(layer.W_h) == 3
        assert len(layer.W_h[0]) == combined
        assert len(layer.b_z) == 3
        assert len(layer.b_r) == 3
        assert len(layer.b_h) == 3
        assert layer.hidden_size == 3

    def test_gru_forward(self):
        rng = create_rng(42)
        layer = create_gru(rng, input_size=4, hidden_size=3)
        h_prev = [0.0, 0.0, 0.0]
        x = [1.0, 2.0, 3.0, 4.0]
        h_new, cache = gru_forward(layer, h_prev, x)
        assert len(h_new) == 3

    def test_gru_forward_deterministic(self):
        rng1 = create_rng(42)
        rng2 = create_rng(42)
        layer1 = create_gru(rng1, input_size=4, hidden_size=3)
        layer2 = create_gru(rng2, input_size=4, hidden_size=3)
        h_prev = [0.0, 0.0, 0.0]
        x = [1.0, 2.0, 3.0, 4.0]
        h1, _ = gru_forward(layer1, h_prev, x)
        h2, _ = gru_forward(layer2, h_prev, x)
        for a, b in zip(h1, h2):
            assert abs(a - b) < 1e-12

    def test_gru_backward(self):
        """Verify GRU gradients against finite differences.

        Uses a non-uniform upstream gradient to catch per-output
        indexing/mixing bugs that uniform [1,1,1] would miss.
        """
        rng = create_rng(42)
        layer = create_gru(rng, input_size=4, hidden_size=3)
        h_prev = [0.1, 0.2, 0.3]
        x = [1.0, 2.0, 3.0, 4.0]
        eps = 1e-5

        h_new, cache = gru_forward(layer, h_prev, x)
        # Non-uniform gradient exercises per-output bookkeeping
        grad_h_new = [1.0, -0.5, 0.3]
        grad_h_prev, grad_x, grad_layer = gru_backward(layer, cache, grad_h_new)

        # Helper: sum of h_new weighted by grad_h_new
        def loss_from_h(h):
            return sum(a * b for a, b in zip(h, grad_h_new))

        # Check grad_x via finite differences
        for i in range(len(x)):
            x_plus = list(x)
            x_plus[i] += eps
            h_plus, _ = gru_forward(layer, h_prev, x_plus)
            x_minus = list(x)
            x_minus[i] -= eps
            h_minus, _ = gru_forward(layer, h_prev, x_minus)
            numerical = (loss_from_h(h_plus) - loss_from_h(h_minus)) / (2 * eps)
            assert abs(numerical - grad_x[i]) < 1e-4, f"grad_x[{i}]: {numerical} vs {grad_x[i]}"

        # Check grad_h_prev via finite differences
        for i in range(len(h_prev)):
            hp_plus = list(h_prev)
            hp_plus[i] += eps
            h_plus, _ = gru_forward(layer, hp_plus, x)
            hp_minus = list(h_prev)
            hp_minus[i] -= eps
            h_minus, _ = gru_forward(layer, hp_minus, x)
            numerical = (loss_from_h(h_plus) - loss_from_h(h_minus)) / (2 * eps)
            assert abs(numerical - grad_h_prev[i]) < 1e-4, f"grad_h_prev[{i}]: {numerical} vs {grad_h_prev[i]}"

        # Check a representative weight gradient (W_z[0][0])
        orig = layer.W_z[0][0]
        layer.W_z[0][0] = orig + eps
        h_plus, _ = gru_forward(layer, h_prev, x)
        layer.W_z[0][0] = orig - eps
        h_minus, _ = gru_forward(layer, h_prev, x)
        layer.W_z[0][0] = orig
        numerical = (loss_from_h(h_plus) - loss_from_h(h_minus)) / (2 * eps)
        assert abs(numerical - grad_layer.W_z[0][0]) < 1e-4, f"grad W_z[0][0]: {numerical} vs {grad_layer.W_z[0][0]}"

        # Check a representative bias gradient (b_h[0])
        orig = layer.b_h[0]
        layer.b_h[0] = orig + eps
        h_plus, _ = gru_forward(layer, h_prev, x)
        layer.b_h[0] = orig - eps
        h_minus, _ = gru_forward(layer, h_prev, x)
        layer.b_h[0] = orig
        numerical = (loss_from_h(h_plus) - loss_from_h(h_minus)) / (2 * eps)
        assert abs(numerical - grad_layer.b_h[0]) < 1e-4, f"grad b_h[0]: {numerical} vs {grad_layer.b_h[0]}"


class TestSGDMomentum:
    def test_sgd_momentum_reduces_loss(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 4, 1], [activations.relu, activations.identity])
        inputs = [1.0, 2.0]
        targets = [5.0]

        output1, _ = network_forward(net, inputs)
        loss1 = mse(output1, targets)

        # Initialize velocities to zero
        velocities = []
        for layer in net.layers:
            rows = len(layer.weights)
            cols = len(layer.weights[0])
            velocities.append(LayerGradients(
                weight_grads=matrix.zeros(rows, cols),
                bias_grads=vector.zeros(len(layer.biases)),
            ))

        for _ in range(20):
            output, cache = network_forward(net, inputs)
            loss_grad = [2.0 * (output[i] - targets[i]) / len(targets) for i in range(len(targets))]
            grads = backward(net, cache, loss_grad)
            velocities = sgd_momentum_update(net, grads, velocities, learning_rate=0.01, momentum=0.9)

        output_final, _ = network_forward(net, inputs)
        loss_final = mse(output_final, targets)
        assert loss_final < loss1


class TestGaussianEntropy:
    def test_gaussian_entropy(self):
        narrow = Gaussian(mean=[0.0], std=[0.5])
        wide = Gaussian(mean=[0.0], std=[2.0])
        assert wide.entropy() > narrow.entropy()


class TestLambdaReturn:
    """Tests for lambda_return.

    Convention: values[k] = V(s_k), next_value = V(s_T).
    rewards[k] is the reward at step k.
    The 1-step return from step k is: r_k + gamma * V(s_{k+1}).
    """

    def test_lambda_return_lambda_zero(self):
        """lambda=0: pure 1-step TD. Bootstrap from V(s_{k+1}) at each step."""
        # 3 steps: s_0 -> s_1 -> s_2 -> s_3
        rewards = [1.0, 2.0, 3.0]
        values = [10.0, 20.0, 30.0]  # V(s_0), V(s_1), V(s_2)
        next_value = 40.0             # V(s_3)
        g = lambda_return(rewards, values, next_value, gamma=0.9, lam=0.0)
        # With lam=0, only the 1-step return matters at each step:
        #   k=2: g_1 = r_2 + gamma*V(s_3) = 3 + 0.9*40 = 39.0
        #   k=1: g_1 = r_1 + gamma*V(s_2) = 2 + 0.9*30 = 29.0
        #   k=0: g_1 = r_0 + gamma*V(s_1) = 1 + 0.9*20 = 19.0
        assert abs(g - 19.0) < 1e-10

    def test_lambda_return_lambda_one(self):
        """lambda=1: Monte Carlo with terminal bootstrap from next_value."""
        rewards = [1.0, 2.0, 3.0]
        values = [10.0, 20.0, 30.0]  # V(s_0), V(s_1), V(s_2)
        next_value = 0.0              # terminal state, no future reward
        g = lambda_return(rewards, values, next_value, gamma=0.9, lam=1.0)
        # With lam=1 and next_value=0, this should equal the MC return:
        #   G = r_0 + gamma*r_1 + gamma^2*r_2 = 1 + 1.8 + 2.43 = 5.23
        mc = discount_return(rewards, gamma=0.9)
        assert abs(g - mc) < 1e-10
        assert abs(g - 5.23) < 1e-10

    def test_lambda_return_matches_n_step(self):
        """With lam=0 and a single step, should match n_step_return."""
        rewards = [5.0]
        values = [100.0]  # V(s_0), irrelevant for 1-step
        next_value = 10.0  # V(s_1)
        g = lambda_return(rewards, values, next_value, gamma=0.9, lam=0.0)
        expected = 5.0 + 0.9 * 10.0  # r + gamma * V(s_1) = 14.0
        assert abs(g - expected) < 1e-10


class TestGaussianPolicy:
    def test_gaussian_policy(self):
        rng = create_rng(42)
        lo, hi = -1.0, 1.0
        for _ in range(50):
            action = gaussian_policy(rng, mean=0.0, std=0.5, lo=lo, hi=hi)
            assert lo <= action <= hi


class TestSoftmaxPolicyError:
    def test_softmax_no_args_raises(self):
        import pytest
        rng = create_rng(42)
        with pytest.raises(TypeError, match="requires q_values"):
            softmax_policy(rng, temperature=1.0)


class TestMetricLog:
    def test_empty_mean(self):
        from policywerk.data.logging import MetricLog
        m = MetricLog(name="test")
        assert m.mean == 0.0

    def test_empty_recent_mean(self):
        from policywerk.data.logging import MetricLog
        m = MetricLog(name="test")
        assert m.recent_mean() == 0.0

    def test_record_and_last(self):
        from policywerk.data.logging import MetricLog
        m = MetricLog(name="test")
        m.record(3.0)
        m.record(5.0)
        assert m.last == 5.0
        assert m.mean == 4.0

    def test_empty_last(self):
        from policywerk.data.logging import MetricLog
        m = MetricLog(name="test")
        assert m.last == 0.0


class TestTrainingLog:
    def test_get_missing_metric(self):
        from policywerk.data.logging import TrainingLog
        log = TrainingLog()
        m = log.get("nonexistent")
        assert m.name == "nonexistent"
        assert m.last == 0.0

    def test_record_and_summary(self):
        from policywerk.data.logging import TrainingLog
        log = TrainingLog()
        log.record("loss", 0.5)
        log.record("loss", 0.3)
        s = log.summary()
        assert "loss=" in s


class TestGradUnknownLoss:
    def test_unknown_loss_raises(self):
        import pytest
        from policywerk.building_blocks.grad import _get_loss_derivative
        with pytest.raises(ValueError, match="Unknown loss"):
            _get_loss_derivative(lambda x, y: 0.0)


class TestGradHuberCheck:
    def test_gradient_check_huber(self):
        """numerical_gradient_check should work with Huber loss."""
        from policywerk.building_blocks.grad import numerical_gradient_check
        from policywerk.building_blocks.network import create_network
        from policywerk.primitives.losses import huber
        rng = create_rng(42)
        net = create_network(rng, [3, 4, 2], [activations.sigmoid, activations.identity])
        max_error = numerical_gradient_check(
            net, [1.0, 0.5, -0.5], [1.0, 0.0], huber
        )
        assert max_error < 1e-4


class TestEpisodeTotalReward:
    def test_episode_total_reward(self):
        from policywerk.building_blocks.mdp import Episode, Transition, State
        s = State(features=[0.0], label="s")
        ep = Episode()
        ep.add(Transition(s, 0, 1.5, s, False))
        ep.add(Transition(s, 0, 2.5, s, True))
        assert ep.total_reward == 4.0


class TestCategoricalLogProb:
    def test_log_prob(self):
        import math
        cat = Categorical(logits=[0.0, 0.0, 0.0])
        # Uniform over 3: each prob = 1/3, log_prob ≈ -1.0986
        lp = cat.log_prob(0)
        assert abs(lp - math.log(1.0 / 3.0)) < 0.01


class TestTracesExtended:
    def test_replace_trace(self):
        trace = EligibilityTrace(gamma=0.9, lam=0.8)
        trace.visit("s1")
        trace.visit("s1")
        assert trace.get("s1") == 2.0  # accumulating
        trace.replace("s1")
        assert trace.get("s1") == 1.0  # replacing resets to 1

    def test_all_traces(self):
        trace = EligibilityTrace(gamma=0.9, lam=0.8)
        trace.visit("s1")
        trace.visit("s2")
        all_t = trace.all_traces()
        assert isinstance(all_t, dict)
        assert "s1" in all_t
        assert "s2" in all_t
        assert all_t["s1"] == 1.0
        assert all_t["s2"] == 1.0
