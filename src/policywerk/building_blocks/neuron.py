"""Level 1: A single neuron.

The fundamental unit of a neural network: activation(dot(weights, inputs) + bias).
Used by the Barto/Sutton ACE/ASE architecture (L02).
"""

from dataclasses import dataclass

from policywerk.primitives import scalar, vector
from policywerk.primitives.random import random_vector

Vector = list[float]


@dataclass
class Neuron:
    weights: Vector  # how much each input matters
    bias: float      # a constant offset added before the activation


def create_neuron(rng, num_inputs: int) -> Neuron:
    weights = random_vector(rng, num_inputs, -1.0, 1.0)
    bias = 0.0
    return Neuron(weights=weights, bias=bias)


def forward(neuron: Neuron, inputs: Vector, activation_fn) -> float:
    """Compute the neuron's output: multiply each input by its weight, add them up, add the bias, then apply the activation function."""
    # pre-activation: weighted sum + bias
    z = scalar.add(vector.dot(neuron.weights, inputs), neuron.bias)
    return activation_fn(z)
