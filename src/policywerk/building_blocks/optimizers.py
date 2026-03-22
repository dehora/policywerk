"""Level 1: Optimizers.

Parameter update rules — SGD, SGD with momentum, and Adam.
Takes gradients and adjusts weights to minimize loss.
"""

from dataclasses import dataclass, field

from policywerk.primitives import scalar, vector, matrix
from policywerk.building_blocks.grad import LayerGradients

Vector = list[float]
Matrix = list[list[float]]


def sgd_update(network, gradients: list[LayerGradients], learning_rate: float):
    """Update weights and biases using vanilla SGD. Modifies network in place."""
    for layer, grads in zip(network.layers, gradients):
        layer.weights = matrix.add(
            layer.weights,
            matrix.scale(-learning_rate, grads.weight_grads),
        )
        layer.biases = vector.add(
            layer.biases,
            vector.scale(-learning_rate, grads.bias_grads),
        )


def sgd_momentum_update(
    network,
    gradients: list[LayerGradients],
    velocities: list[LayerGradients],
    learning_rate: float,
    momentum: float = 0.9,
) -> list[LayerGradients]:
    """Update weights using SGD with momentum. Returns updated velocities."""
    new_velocities = []

    for layer, grads, vel in zip(network.layers, gradients, velocities):
        new_weight_vel = matrix.add(
            matrix.scale(momentum, vel.weight_grads),
            matrix.scale(-learning_rate, grads.weight_grads),
        )
        new_bias_vel = vector.add(
            vector.scale(momentum, vel.bias_grads),
            vector.scale(-learning_rate, grads.bias_grads),
        )

        layer.weights = matrix.add(layer.weights, new_weight_vel)
        layer.biases = vector.add(layer.biases, new_bias_vel)

        new_velocities.append(LayerGradients(
            weight_grads=new_weight_vel,
            bias_grads=new_bias_vel,
        ))

    return new_velocities


@dataclass
class AdamState:
    """Per-layer Adam optimizer state."""
    m_weights: Matrix    # first moment (mean of gradients)
    v_weights: Matrix    # second moment (mean of squared gradients)
    m_biases: Vector
    v_biases: Vector
    t: int = 0           # timestep


def create_adam_state(network) -> list[AdamState]:
    """Initialize Adam state for each layer."""
    states = []
    for layer in network.layers:
        rows = len(layer.weights)
        cols = len(layer.weights[0])
        states.append(AdamState(
            m_weights=matrix.zeros(rows, cols),
            v_weights=matrix.zeros(rows, cols),
            m_biases=vector.zeros(len(layer.biases)),
            v_biases=vector.zeros(len(layer.biases)),
        ))
    return states


def adam_update(
    network,
    gradients: list[LayerGradients],
    states: list[AdamState],
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    """Update weights using Adam optimizer. Modifies network and states in place.

    Adam combines momentum (first moment) with RMSProp (second moment)
    and includes bias correction for the early timesteps.
    """
    for layer, grads, state in zip(network.layers, gradients, states):
        state.t += 1

        # Update first moment: m = β₁m + (1-β₁)g
        for r in range(len(layer.weights)):
            for c in range(len(layer.weights[0])):
                state.m_weights[r][c] = scalar.add(
                    scalar.multiply(beta1, state.m_weights[r][c]),
                    scalar.multiply(1.0 - beta1, grads.weight_grads[r][c]),
                )
                state.v_weights[r][c] = scalar.add(
                    scalar.multiply(beta2, state.v_weights[r][c]),
                    scalar.multiply(1.0 - beta2, scalar.multiply(
                        grads.weight_grads[r][c], grads.weight_grads[r][c])),
                )

        for b in range(len(layer.biases)):
            state.m_biases[b] = scalar.add(
                scalar.multiply(beta1, state.m_biases[b]),
                scalar.multiply(1.0 - beta1, grads.bias_grads[b]),
            )
            state.v_biases[b] = scalar.add(
                scalar.multiply(beta2, state.v_biases[b]),
                scalar.multiply(1.0 - beta2, scalar.multiply(
                    grads.bias_grads[b], grads.bias_grads[b])),
            )

        # Bias correction
        bc1 = scalar.inverse(scalar.subtract(1.0, scalar.power(beta1, state.t)))
        bc2 = scalar.inverse(scalar.subtract(1.0, scalar.power(beta2, state.t)))

        # Apply updates: w -= lr * m_hat / (sqrt(v_hat) + eps)
        for r in range(len(layer.weights)):
            for c in range(len(layer.weights[0])):
                m_hat = scalar.multiply(state.m_weights[r][c], bc1)
                v_hat = scalar.multiply(state.v_weights[r][c], bc2)
                update = scalar.multiply(
                    learning_rate,
                    scalar.multiply(m_hat, scalar.inverse(scalar.add(scalar.power(v_hat, 0.5), eps))),
                )
                layer.weights[r][c] = scalar.subtract(layer.weights[r][c], update)

        for b in range(len(layer.biases)):
            m_hat = scalar.multiply(state.m_biases[b], bc1)
            v_hat = scalar.multiply(state.v_biases[b], bc2)
            update = scalar.multiply(
                learning_rate,
                scalar.multiply(m_hat, scalar.inverse(scalar.add(scalar.power(v_hat, 0.5), eps))),
            )
            layer.biases[b] = scalar.subtract(layer.biases[b], update)
