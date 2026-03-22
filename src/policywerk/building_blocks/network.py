"""Level 1: Network container.

A sequential collection of dense layers. Passes input through each layer in
sequence — each layer's output becomes the next layer's input. Intermediate
values are cached for backprop.
"""

from dataclasses import dataclass

from policywerk.building_blocks.dense import DenseLayer, DenseCache, create_dense, dense_forward

Vector = list[float]


@dataclass
class Network:
    layers: list[DenseLayer]
    activation_fns: list     # one non-linear function per layer, applied after each layer's linear computation


@dataclass
class NetworkCache:
    layer_caches: list[DenseCache]


def create_network(rng, layer_sizes: list[int], activation_fns: list) -> Network:
    """Create a sequential network.

    layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
    activation_fns: one per layer (len = len(layer_sizes) - 1)
    """
    layers = []
    for layer_idx in range(len(layer_sizes) - 1):
        layer = create_dense(rng, layer_sizes[layer_idx], layer_sizes[layer_idx + 1])
        layers.append(layer)
    return Network(layers=layers, activation_fns=activation_fns)


def network_forward(network: Network, inputs: Vector) -> tuple[Vector, NetworkCache]:
    """Thread input through every layer, caching each step."""
    layer_caches = []
    current = inputs
    for layer, activation_fn in zip(network.layers, network.activation_fns):
        current, cache = dense_forward(layer, current, activation_fn)
        layer_caches.append(cache)
    return current, NetworkCache(layer_caches=layer_caches)
