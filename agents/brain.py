"""
Neural networks for agent decision making, optimized with JAX.
Includes both individual network handling and batch processing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from functools import partial
from typing import List, Tuple, Dict, Optional, Union

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG, NN_CONFIG


class NeuralNetwork:
    """Neural network for agent decision making, optimized with JAX."""

    def __init__(self, layer_sizes=None, weights=None, key=None):
        """
        Initialize neural network with optional pre-defined weights.

        Args:
            layer_sizes: List of layer sizes (input, hidden, output)
            weights: List of weight matrices if loading existing network
            key: JAX random key for initialization
        """
        if layer_sizes is None:
            layer_sizes = NN_CONFIG["layer_sizes"]

        self.layer_sizes = layer_sizes

        # Get activation functions
        activation_names = NN_CONFIG["activation_funcs"]
        self.activations = []
        for name in activation_names:
            if name == "relu":
                self.activations.append(jax.nn.relu)
            elif name == "sigmoid":
                self.activations.append(jax.nn.sigmoid)
            elif name == "tanh":
                self.activations.append(jnp.tanh)
            else:
                raise ValueError(f"Unknown activation function: {name}")

        # Initialize weights randomly if not provided
        if weights is None:
            if key is None:
                key = random.PRNGKey(np.random.randint(0, 1000000))

            self.weights = self._init_weights(key)
        else:
            self.weights = weights

        # JIT compile forward function for performance
        self.forward_jit = jit(self._forward_impl)

        # JIT compile batch forward function
        self.batch_forward_jit = jit(vmap(self._forward_impl, in_axes=(None, 0)))

        # Compile gradient functions
        self._create_gradient_functions()

    def _init_weights(self, key):
        """Initialize weights for the neural network."""
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            key, subkey = random.split(key)

            # Initialize weights with scaled random values
            scale = NN_CONFIG["weight_init_range"]
            w = scale * random.normal(
                subkey,
                (self.layer_sizes[i], self.layer_sizes[i + 1])
            )

            # Add bias terms
            key, subkey = random.split(key)
            b = NN_CONFIG["bias_init_range"] * random.normal(
                subkey,
                (self.layer_sizes[i + 1],)
            )

            weights.append((w, b))

        return weights

    def _forward_impl(self, weights, x):
        """
        Implement forward pass through the network.

        Args:
            weights: List of (W, b) tuples
            x: Input vector

        Returns:
            Output vector
        """
        activations = x

        # Process each layer
        for i, ((W, b), activation_fn) in enumerate(zip(weights, self.activations)):
            outputs = jnp.dot(activations, W) + b
            activations = activation_fn(outputs)

        return activations

    def predict(self, x):
        """
        Make a prediction with numpy input/output for compatibility.

        Args:
            x: Input vector or batch of vectors

        Returns:
            Prediction vector or batch of vectors
        """
        x_jax = jnp.asarray(x)

        # Handle batch input
        if x_jax.ndim > 1 and x_jax.shape[1] == self.layer_sizes[0]:
            return np.asarray(self.batch_forward_jit(self.weights, x_jax))

        # Handle single input
        return np.asarray(self.forward_jit(self.weights, x_jax))

    def _create_gradient_functions(self):
        """Create functions for gradient-based learning."""

        # Define loss function (MSE for now)
        def loss_fn(weights, x, y):
            pred = self._forward_impl(weights, x)
            return jnp.mean((pred - y) ** 2)

        # Create gradient function
        self._grad_fn = jit(grad(loss_fn))

        # Create batched gradient function
        self._batch_grad_fn = jit(vmap(grad(loss_fn), in_axes=(None, 0, 0)))

    def update_weights(self, x, y, learning_rate=None):
        """
        Update weights using gradient descent.

        Args:
            x: Input data
            y: Target outputs
            learning_rate: Learning rate (optional)

        Returns:
            Loss value
        """
        if learning_rate is None:
            learning_rate = NN_CONFIG["learning_rate"]

        x_jax = jnp.asarray(x)
        y_jax = jnp.asarray(y)

        # Calculate gradients
        if x_jax.ndim > 1 and y_jax.ndim > 1:
            # Batch update
            grads = self._batch_grad_fn(self.weights, x_jax, y_jax)
            # Average gradients
            avg_grads = jax.tree_map(lambda g: jnp.mean(g, axis=0), grads)
            # Update weights
            self.weights = [(w - learning_rate * dw, b - learning_rate * db)
                            for (w, b), (dw, db) in zip(self.weights, avg_grads)]
        else:
            # Single example update
            grads = self._grad_fn(self.weights, x_jax, y_jax)
            # Update weights
            self.weights = [(w - learning_rate * dw, b - learning_rate * db)
                            for (w, b), (dw, db) in zip(self.weights, grads)]

        # Calculate loss
        loss = jnp.mean((self.predict(x) - y) ** 2)
        return loss

    def mutate(self, mutation_rate=None, mutation_scale=0.2):
        """
        Mutate the weights of the neural network.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_scale: Scale of mutations

        Returns:
            New neural network with mutated weights
        """
        if mutation_rate is None:
            mutation_rate = CONFIG["mutation_rate"]

        # Create mutation masks and values
        new_weights = []
        for w, b in self.weights:
            # Mutate weights
            mask_w = np.random.random(w.shape) < mutation_rate
            mutations_w = np.random.normal(0, mutation_scale, w.shape) * mask_w

            # Mutate biases
            mask_b = np.random.random(b.shape) < mutation_rate
            mutations_b = np.random.normal(0, mutation_scale, b.shape) * mask_b

            # Apply mutations
            new_w = np.array(w) + mutations_w
            new_b = np.array(b) + mutations_b

            new_weights.append((new_w, new_b))

        # Create a new network with mutated weights
        return NeuralNetwork(self.layer_sizes, new_weights)

    def crossover(self, other, crossover_points=None):
        """
        Perform crossover with another neural network.

        Args:
            other: Other neural network for crossover
            crossover_points: Number of crossover points (defaults to 1 per layer)

        Returns:
            New neural network with crossed-over weights
        """
        if not isinstance(other, NeuralNetwork):
            raise TypeError("Crossover partner must be a NeuralNetwork")

        if self.layer_sizes != other.layer_sizes:
            raise ValueError("Networks must have the same architecture for crossover")

        if crossover_points is None:
            crossover_points = [1] * len(self.weights)

        # Create new weights through crossover
        new_weights = []
        for i, ((w1, b1), (w2, b2)) in enumerate(zip(self.weights, other.weights)):
            # For weight matrices
            if np.random.random() < 0.5:
                # Row-wise crossover
                rows = w1.shape[0]
                crossover_row = np.random.randint(1, rows) if rows > 1 else 0
                new_w = np.vstack([w1[:crossover_row], w2[crossover_row:]])
            else:
                # Element-wise crossover
                mask = np.random.random(w1.shape) < 0.5
                new_w = w1 * mask + w2 * ~mask

            # For bias vectors
            mask_b = np.random.random(b1.shape) < 0.5
            new_b = b1 * mask_b + b2 * ~mask_b

            new_weights.append((new_w, new_b))

        # Create a new network with crossed-over weights
        return NeuralNetwork(self.layer_sizes, new_weights)

    def get_weights_flat(self):
        """
        Get weights as a flat array for compatibility with DEAP.

        Returns:
            Flattened weights array
        """
        flat_weights = []
        for w, b in self.weights:
            flat_weights.extend(w.flatten())
            flat_weights.extend(b.flatten())
        return np.array(flat_weights)

    def set_weights_flat(self, flat_weights):
        """
        Set weights from a flat array (for DEAP compatibility).

        Args:
            flat_weights: Flattened weights array

        Returns:
            Self (for chaining)
        """
        idx = 0
        new_weights = []

        for i in range(len(self.layer_sizes) - 1):
            # Extract weight matrix
            w_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            w_size = w_shape[0] * w_shape[1]
            w = flat_weights[idx:idx + w_size].reshape(w_shape)
            idx += w_size

            # Extract bias vector
            b_size = self.layer_sizes[i + 1]
            b = flat_weights[idx:idx + b_size]
            idx += b_size

            new_weights.append((w, b))

        self.weights = [(jnp.array(w), jnp.array(b)) for w, b in new_weights]

        # Recompile JIT functions with new weights
        self.forward_jit = jit(partial(self._forward_impl))
        self.batch_forward_jit = jit(vmap(partial(self._forward_impl), in_axes=(None, 0)))

        return self

    def clone(self):
        """
        Create a copy of this neural network.

        Returns:
            New neural network with the same weights
        """
        new_weights = [(np.array(w), np.array(b)) for w, b in self.weights]
        return NeuralNetwork(self.layer_sizes, new_weights)


# Batch processing functions for efficient updates
def batch_predict(networks, inputs):
    """
    Perform batch prediction for multiple networks and inputs.

    Args:
        networks: List of NeuralNetwork objects
        inputs: Batch of input vectors

    Returns:
        Batch of predictions
    """
    # Convert inputs to JAX array
    inputs_jax = jnp.asarray(inputs)

    # Get weights from all networks
    all_weights = [network.weights for network in networks]

    # Define batched forward function
    @jit
    def batched_forward(weights_list, x_batch):
        def forward_single(weights, x):
            # This should match the _forward_impl method in NeuralNetwork
            activations = x
            for i, ((W, b), activation_fn) in enumerate(zip(
                    weights, [jax.nn.relu, jax.nn.sigmoid])):
                outputs = jnp.dot(activations, W) + b
                activations = activation_fn(outputs)
            return activations

        # Map over batch dimension
        return vmap(forward_single, in_axes=(0, 0))(weights_list, x_batch)

    # Perform batch prediction
    return np.asarray(batched_forward(all_weights, inputs_jax))


# Memory-efficient JAX functions
def create_memory_efficient_network(layer_sizes=None, key=None):
    """
    Create a memory-efficient neural network using JAX functional style.

    Args:
        layer_sizes: List of layer sizes
        key: JAX random key

    Returns:
        Tuple of (params, forward_fn, predict_fn)
    """
    if layer_sizes is None:
        layer_sizes = NN_CONFIG["layer_sizes"]

    if key is None:
        key = random.PRNGKey(np.random.randint(0, 1000000))

    # Initialize parameters
    def init_layer(key, in_size, out_size):
        scale = NN_CONFIG["weight_init_range"]
        key1, key2 = random.split(key)
        w = scale * random.normal(key1, (in_size, out_size))
        b = NN_CONFIG["bias_init_range"] * random.normal(key2, (out_size,))
        return w, b

    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = random.split(key)
        params.append(init_layer(subkey, layer_sizes[i], layer_sizes[i + 1]))

    # Define forward function
    @jit
    def forward_fn(params, x):
        for i, ((w, b), activation_name) in enumerate(zip(params, NN_CONFIG["activation_funcs"])):
            x = jnp.dot(x, w) + b
            if activation_name == "relu":
                x = jax.nn.relu(x)
            elif activation_name == "sigmoid":
                x = jax.nn.sigmoid(x)
            elif activation_name == "tanh":
                x = jnp.tanh(x)
        return x

    # Define prediction function
    def predict_fn(params, x):
        x_jax = jnp.asarray(x)
        return np.asarray(forward_fn(params, x_jax))

    return params, forward_fn, predict_fn