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
    weights: Vector
    bias: float


def create_neuron(rng, num_inputs: int) -> Neuron:
    weights = random_vector(rng, num_inputs, -1.0, 1.0)
    bias = 0.0
    return Neuron(weights=weights, bias=bias)


def forward(neuron: Neuron, inputs: Vector, activation_fn) -> float:
    z = scalar.add(vector.dot(neuron.weights, inputs), neuron.bias)
    return activation_fn(z)
