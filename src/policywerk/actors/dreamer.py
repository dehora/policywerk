"""Level 3: DreamerV3 World Model (simplified).

Hafner et al. (2023), 'Mastering Diverse Domains through World Models.'

PPO (L06) learns from real experience only. Every training step
requires running the real environment. DreamerV3 takes a different
approach: learn a model of the world, then imagine trajectories
and train the policy on imagined data.

The world model has four parts:

  1. Encoder: compresses 256-pixel observations into a 32-dim
     latent representation.

  2. GRU dynamics: given the current latent state and an action,
     predicts the next latent state. This is the "physics engine"
     the agent learns from experience.

  3. Decoder: reconstructs pixels from the latent state. Provides
     a training signal that forces the encoder to preserve useful
     information.

  4. Reward head: predicts reward from the latent state. Provides
     a training signal that forces the dynamics model to track
     reward-relevant features.

Training alternates between two phases: (a) train the world model
on real data, and (b) imagine trajectories in latent space and
train the actor-critic on imagined data.

This is a simplified version of DreamerV3. The full paper uses a
stochastic RSSM with KL balancing, symlog predictions, twohot
value distributions, and several other stabilization tricks. We
keep the core idea—learn dynamics, train in imagination—and use
MSE losses with teacher forcing for simplicity.
"""

import math

from policywerk.primitives import scalar, vector, matrix
from policywerk.primitives.progress import progress_bar, progress_done
from policywerk.primitives.activations import tanh_, identity, sigmoid
from policywerk.primitives.losses import mse_derivative
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.network import Network, create_network, network_forward
from policywerk.building_blocks.grad import backward, backward_with_input_grad, LayerGradients
from policywerk.building_blocks.optimizers import create_adam_state, adam_update
from policywerk.building_blocks.recurrent import create_gru, gru_forward, gru_backward, GRULayer
from policywerk.building_blocks.distributions import Gaussian
from policywerk.building_blocks.returns import lambda_return

Vector = list[float]


def _zeros(n: int) -> Vector:
    return [0.0] * n


def _add_grads(acc: list[LayerGradients], new: list[LayerGradients]) -> None:
    """Accumulate gradients in place."""
    for i, g in enumerate(new):
        acc[i].weight_grads = matrix.add(acc[i].weight_grads, g.weight_grads)
        acc[i].bias_grads = vector.add(acc[i].bias_grads, g.bias_grads)


def _zero_grads(network: Network) -> list[LayerGradients]:
    """Create zeroed gradient accumulators for a network."""
    acc = []
    for layer in network.layers:
        rows = len(layer.weights)
        cols = len(layer.weights[0])
        acc.append(LayerGradients(
            weight_grads=matrix.zeros(rows, cols),
            bias_grads=vector.zeros(len(layer.biases)),
        ))
    return acc


def _scale_grads(grads: list[LayerGradients], factor: float) -> None:
    """Scale gradient accumulators in place."""
    for g in grads:
        g.weight_grads = matrix.scale(factor, g.weight_grads)
        g.bias_grads = vector.scale(factor, g.bias_grads)


def dreamer(
    env,
    num_iterations: int = 100,
    steps_per_iter: int = 200,
    world_model_epochs: int = 5,
    imagination_horizon: int = 15,
    num_imaginations: int = 16,
    gamma: float = 0.99,
    lam: float = 0.95,
    learning_rate_wm: float = 0.001,
    learning_rate_actor: float = 0.0005,
    learning_rate_critic: float = 0.001,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    seed: int = 42,
) -> tuple[dict, list[dict]]:
    """Train a Dreamer agent: learn a world model, then train in imagination.

    Returns (networks_dict, history) where networks_dict contains all
    trained components and each history entry has:
      iteration, avg_reward, recon_loss, reward_loss, imagined_reward
    """
    rng = create_rng(seed)

    # ---- Create world model ----
    # Encoder: 256 pixels → latent_dim
    encoder = create_network(rng, [256, hidden_dim, latent_dim], [tanh_, tanh_])
    # GRU dynamics: (latent_state, action) → next_latent_state
    # input_size = 2 (action is [fx, fy])
    gru = create_gru(rng, input_size=2, hidden_size=latent_dim)
    # Decoder: latent_dim → 256 pixels (sigmoid for [0,1] pixel range)
    decoder = create_network(rng, [latent_dim, hidden_dim, 256], [tanh_, sigmoid])
    # Reward head: latent_dim → scalar reward
    reward_head = create_network(rng, [latent_dim, 32, 1], [tanh_, identity])

    # ---- Create actor-critic (operate in latent space) ----
    # Actor: latent → [mean_x, mean_y, log_std_x, log_std_y]
    actor = create_network(rng, [latent_dim, 32, 4], [tanh_, identity])
    # Critic: latent → scalar value
    critic = create_network(rng, [latent_dim, 32, 1], [tanh_, identity])

    # Optimizers
    enc_adam = create_adam_state(encoder)
    dec_adam = create_adam_state(decoder)
    rew_adam = create_adam_state(reward_head)
    act_adam = create_adam_state(actor)
    crt_adam = create_adam_state(critic)
    # GRU Adam state (manual—GRU isn't a Network, so we track m/v per weight)
    _gru_attrs = ["W_z", "b_z", "W_r", "b_r", "W_h", "b_h"]
    gru_m: dict = {}
    gru_v: dict = {}
    for attr in _gru_attrs:
        param = getattr(gru, attr)
        if isinstance(param[0], list):  # matrix
            gru_m[attr] = [[0.0] * len(param[0]) for _ in range(len(param))]
            gru_v[attr] = [[0.0] * len(param[0]) for _ in range(len(param))]
        else:  # vector
            gru_m[attr] = [0.0] * len(param)
            gru_v[attr] = [0.0] * len(param)
    gru_t = [0]

    history: list[dict] = []
    state = env.reset()
    prev_action = [0.0, 0.0]

    for iteration in range(num_iterations):
        # ---- Phase 1: Collect T real steps ----
        pixels_seq: list[Vector] = []
        actions_seq: list[Vector] = []
        rewards_seq: list[float] = []
        dones_seq: list[bool] = []

        ep_rewards: list[float] = []
        ep_reward = 0.0

        for _t in range(steps_per_iter):
            pixels_seq.append(list(state.features))

            # Encode observation and choose action
            z, _ = network_forward(encoder, state.features)
            actor_out, _ = network_forward(actor, z)
            mean_x, mean_y = actor_out[0], actor_out[1]
            log_std_x = scalar.clamp(actor_out[2], -2.0, 2.0)
            log_std_y = scalar.clamp(actor_out[3], -2.0, 2.0)
            std_x, std_y = scalar.exp(log_std_x), scalar.exp(log_std_y)

            dist = Gaussian([mean_x, mean_y], [std_x, std_y])
            raw_action = dist.sample(rng)
            action = [scalar.clamp(raw_action[0], -1.0, 1.0),
                      scalar.clamp(raw_action[1], -1.0, 1.0)]

            next_state, reward, done = env.step_continuous(action)
            ep_reward += reward

            actions_seq.append(list(action))
            rewards_seq.append(reward)
            dones_seq.append(done)

            state = next_state
            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                state = env.reset()

        # ---- Phase 2: Train world model (teacher-forced) ----
        iter_recon_loss = 0.0
        iter_reward_loss = 0.0

        for _epoch in range(world_model_epochs):
            enc_acc = _zero_grads(encoder)
            dec_acc = _zero_grads(decoder)
            rew_acc = _zero_grads(reward_head)
            gru_w_acc = None  # accumulated GRU weight gradients

            for t in range(steps_per_iter - 1):
                # Teacher forcing: encode the actual observation
                z_t, enc_cache = network_forward(encoder, pixels_seq[t])

                # GRU step: predict next latent from (current_latent, action)
                h_next, gru_cache = gru_forward(gru, z_t, actions_seq[t])

                # Decoder: predict next pixels from h_next
                pixel_pred, dec_cache = network_forward(decoder, h_next)
                # Reward head: predict reward from h_next
                reward_pred, rew_cache = network_forward(reward_head, h_next)

                # Losses (MSE against actual next observation and reward)
                recon_grad = mse_derivative(pixel_pred, pixels_seq[t + 1])
                reward_grad = mse_derivative(reward_pred, [rewards_seq[t]])

                # Track losses
                recon_err = sum((p - a) ** 2 for p, a in
                                zip(pixel_pred, pixels_seq[t + 1])) / 256.0
                reward_err = (reward_pred[0] - rewards_seq[t]) ** 2
                iter_recon_loss += recon_err
                iter_reward_loss += reward_err

                # ---- Backprop through the world model graph ----
                # Decoder backward → grad_h from reconstruction
                dec_grads, grad_h_dec = backward_with_input_grad(
                    decoder, dec_cache, recon_grad)
                # Reward head backward → grad_h from reward prediction
                rew_grads, grad_h_rew = backward_with_input_grad(
                    reward_head, rew_cache, reward_grad)
                # Total gradient on h_next
                grad_h = vector.add(grad_h_dec, grad_h_rew)

                # GRU backward: grad on hidden → grads for weights, z_t, action
                grad_z, grad_action, gru_grads = gru_backward(
                    gru, gru_cache, grad_h)

                # Encoder backward
                enc_grads = backward(encoder, enc_cache, grad_z)

                # Accumulate all gradients
                _add_grads(enc_acc, enc_grads)
                _add_grads(dec_acc, dec_grads)
                _add_grads(rew_acc, rew_grads)

                # Accumulate GRU gradients
                if gru_w_acc is None:
                    gru_w_acc = gru_grads
                else:
                    for attr in ["W_z", "b_z", "W_r", "b_r", "W_h", "b_h"]:
                        old = getattr(gru_w_acc, attr)
                        new = getattr(gru_grads, attr)
                        if isinstance(old[0], list):  # matrix
                            setattr(gru_w_acc, attr, matrix.add(old, new))
                        else:  # vector
                            setattr(gru_w_acc, attr, vector.add(old, new))

            # Average and update
            n_steps = float(steps_per_iter - 1)
            inv_n = scalar.inverse(n_steps)
            _scale_grads(enc_acc, inv_n)
            _scale_grads(dec_acc, inv_n)
            _scale_grads(rew_acc, inv_n)

            adam_update(encoder, enc_acc, enc_adam, learning_rate_wm)
            adam_update(decoder, dec_acc, dec_adam, learning_rate_wm)
            adam_update(reward_head, rew_acc, rew_adam, learning_rate_wm)

            # GRU Adam update (manual—GRU isn't a Network)
            if gru_w_acc is not None:
                gru_t[0] += 1
                for attr in ["W_z", "b_z", "W_r", "b_r", "W_h", "b_h"]:
                    param = getattr(gru, attr)
                    grad = getattr(gru_w_acc, attr)
                    m = gru_m[attr]
                    v = gru_v[attr]
                    if isinstance(param[0], list):  # matrix
                        for r in range(len(param)):
                            for c in range(len(param[0])):
                                g = grad[r][c] / n_steps
                                m[r][c] = 0.9 * m[r][c] + 0.1 * g
                                v[r][c] = 0.999 * v[r][c] + 0.001 * g * g
                                m_hat = m[r][c] / (1 - 0.9 ** gru_t[0])
                                v_hat = v[r][c] / (1 - 0.999 ** gru_t[0])
                                param[r][c] -= learning_rate_wm * m_hat / (math.sqrt(v_hat) + 1e-8)
                    else:  # vector (bias)
                        for i in range(len(param)):
                            g = grad[i] / n_steps
                            m[i] = 0.9 * m[i] + 0.1 * g
                            v[i] = 0.999 * v[i] + 0.001 * g * g
                            m_hat = m[i] / (1 - 0.9 ** gru_t[0])
                            v_hat = v[i] / (1 - 0.999 ** gru_t[0])
                            param[i] -= learning_rate_wm * m_hat / (math.sqrt(v_hat) + 1e-8)

        # ---- Phase 3: Imagine trajectories ----
        all_imagined_h: list[list[Vector]] = []
        all_imagined_actions: list[list[Vector]] = []
        all_imagined_rewards: list[list[float]] = []
        all_imagined_values: list[list[float]] = []
        all_imagined_log_probs: list[list[float]] = []

        for _traj in range(num_imaginations):
            # Start from a random real observation
            start_idx = rng.randint(0, len(pixels_seq) - 1)
            h, _ = network_forward(encoder, pixels_seq[start_idx])

            traj_h, traj_a, traj_r, traj_v, traj_lp = [], [], [], [], []

            for _step in range(imagination_horizon):
                traj_h.append(list(h))

                # Actor chooses action
                actor_out, _ = network_forward(actor, h)
                mean_x, mean_y = actor_out[0], actor_out[1]
                log_std_x = scalar.clamp(actor_out[2], -2.0, 2.0)
                log_std_y = scalar.clamp(actor_out[3], -2.0, 2.0)
                std_x, std_y = scalar.exp(log_std_x), scalar.exp(log_std_y)
                dist = Gaussian([mean_x, mean_y], [std_x, std_y])
                raw_a = dist.sample(rng)
                action = [scalar.clamp(raw_a[0], -1.0, 1.0),
                          scalar.clamp(raw_a[1], -1.0, 1.0)]
                lp = dist.log_prob(raw_a)

                # Critic estimates value
                value_out, _ = network_forward(critic, h)
                traj_v.append(value_out[0])

                # GRU dynamics: predict next state
                h, _ = gru_forward(gru, h, action)

                # Reward prediction
                rew_out, _ = network_forward(reward_head, h)
                traj_r.append(rew_out[0])
                traj_a.append(action)
                traj_lp.append(lp)

            all_imagined_h.append(traj_h)
            all_imagined_actions.append(traj_a)
            all_imagined_rewards.append(traj_r)
            all_imagined_values.append(traj_v)
            all_imagined_log_probs.append(traj_lp)

        # ---- Phase 4: Train actor-critic on imagined data ----
        act_acc = _zero_grads(actor)
        crt_acc = _zero_grads(critic)
        total_imagined_reward = 0.0
        total_ac_steps = 0

        for traj_idx in range(num_imaginations):
            rewards_i = all_imagined_rewards[traj_idx]
            values_i = all_imagined_values[traj_idx]
            h_seq = all_imagined_h[traj_idx]
            a_seq = all_imagined_actions[traj_idx]

            total_imagined_reward += sum(rewards_i)

            # Bootstrap value from the final imagined state
            final_h = list(all_imagined_h[traj_idx][-1])
            # Step GRU one more time for the terminal state value
            h_final, _ = gru_forward(gru, final_h, a_seq[-1])
            v_final_out, _ = network_forward(critic, h_final)
            next_value = v_final_out[0]

            # Compute lambda returns
            returns_i = []
            for t in range(len(rewards_i)):
                r_slice = rewards_i[t:]
                v_slice = values_i[t:]
                ret = lambda_return(r_slice, v_slice, next_value, gamma, lam)
                returns_i.append(ret)

            # Advantages
            advantages = [ret - val for ret, val in zip(returns_i, values_i)]
            # Normalize
            if len(advantages) > 1:
                adv_mean = sum(advantages) / len(advantages)
                adv_var = sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)
                adv_std = math.sqrt(adv_var) + 1e-8
                advantages = [(a - adv_mean) / adv_std for a in advantages]

            for t in range(len(rewards_i)):
                h_t = h_seq[t]

                # Actor gradient: -advantage * d_log_prob / d_params
                actor_out, actor_cache = network_forward(actor, h_t)
                m_x, m_y = actor_out[0], actor_out[1]
                ls_x = scalar.clamp(actor_out[2], -2.0, 2.0)
                ls_y = scalar.clamp(actor_out[3], -2.0, 2.0)
                s_x, s_y = scalar.exp(ls_x), scalar.exp(ls_y)

                # d_log_prob / d_output for each of the 4 outputs
                a_x, a_y = a_seq[t][0], a_seq[t][1]
                dlp_dm_x = (a_x - m_x) / (s_x * s_x)
                dlp_dm_y = (a_y - m_y) / (s_y * s_y)
                z_x = (a_x - m_x) / s_x
                z_y = (a_y - m_y) / s_y
                dlp_dls_x = z_x * z_x - 1.0
                dlp_dls_y = z_y * z_y - 1.0

                adv = advantages[t]
                actor_loss_grad = [
                    -adv * dlp_dm_x,
                    -adv * dlp_dm_y,
                    -adv * dlp_dls_x,
                    -adv * dlp_dls_y,
                ]
                actor_grads = backward(actor, actor_cache, actor_loss_grad)
                _add_grads(act_acc, actor_grads)

                # Critic gradient: MSE toward lambda return
                value_out, critic_cache = network_forward(critic, h_t)
                critic_grad = mse_derivative(value_out, [returns_i[t]])
                critic_grads = backward(critic, critic_cache, critic_grad)
                _add_grads(crt_acc, critic_grads)

                total_ac_steps += 1

        # Average and update actor-critic
        if total_ac_steps > 0:
            inv_ac = scalar.inverse(float(total_ac_steps))
            _scale_grads(act_acc, inv_ac)
            _scale_grads(crt_acc, inv_ac)
            adam_update(actor, act_acc, act_adam, learning_rate_actor)
            adam_update(critic, crt_acc, crt_adam, learning_rate_critic)

        # ---- Record history ----
        avg_reward = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
        n_wm = float((steps_per_iter - 1) * world_model_epochs)
        avg_recon = iter_recon_loss / n_wm if n_wm > 0 else 0.0
        avg_rew_loss = iter_reward_loss / n_wm if n_wm > 0 else 0.0
        avg_imagined = total_imagined_reward / num_imaginations if num_imaginations > 0 else 0.0

        history.append({
            "iteration": iteration,
            "avg_reward": avg_reward,
            "recon_loss": avg_recon,
            "reward_loss": avg_rew_loss,
            "imagined_reward": avg_imagined,
        })

        progress_bar(
            iteration + 1, num_iterations,
            f"reward={avg_reward:+7.1f}  recon={avg_recon:.4f}",
        )

    progress_done()

    networks = {
        "encoder": encoder,
        "gru": gru,
        "decoder": decoder,
        "reward_head": reward_head,
        "actor": actor,
        "critic": critic,
    }
    return networks, history
