"""
network.py -- a small neural network trained entirely by hand: the
forward pass, the backward pass (backpropagation) and the weight
update are all written out explicitly below, matching the proposal's
design challenge "a neural network will be designed and trained from
first principles" (section 2.1). Two of these are used elsewhere (in
main.py): one recognises WHICH character was written, one recognises
WHICH user wrote it.

Only numpy is used -- it is a maths library (matrix multiplication),
not a neural-network library. No autograd, no layers-as-objects.
Weights are just a plain dict of arrays, not a class, so it stays easy
to read and to rewrite.
"""
import numpy as np


def init_weights(input_size, hidden_size, output_size, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.1, (input_size, hidden_size)),
        "b1": np.zeros(hidden_size),
        "W2": rng.normal(0, 0.1, (hidden_size, output_size)),
        "b2": np.zeros(output_size),
    }


def _relu(x):
    return np.maximum(0, x)


def _softmax(x):
    shifted = x - x.max(axis=-1, keepdims=True)  # for numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def forward(weights, x):
    """x: (batch, input_size) flattened 0/1 character images.
    One hidden layer (ReLU), one output layer (softmax)."""
    z1 = x @ weights["W1"] + weights["b1"]
    a1 = _relu(z1)
    z2 = a1 @ weights["W2"] + weights["b2"]
    probs = _softmax(z2)
    cache = (x, z1, a1)
    return probs, cache


def backward(weights, cache, probs, targets, learning_rate=0.05):
    """One gradient-descent step. targets: integer class label per
    row of x. For softmax + cross-entropy the output-layer gradient
    simplifies neatly to (predicted - actual)."""
    x, z1, a1 = cache
    n = x.shape[0]

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), targets] = 1
    d_output = (probs - one_hot) / n

    d_W2 = a1.T @ d_output
    d_b2 = d_output.sum(axis=0)

    d_hidden = d_output @ weights["W2"].T
    d_hidden[z1 <= 0] = 0  # ReLU derivative: 0 where the input was negative

    d_W1 = x.T @ d_hidden
    d_b1 = d_hidden.sum(axis=0)

    weights["W2"] -= learning_rate * d_W2
    weights["b2"] -= learning_rate * d_b2
    weights["W1"] -= learning_rate * d_W1
    weights["b1"] -= learning_rate * d_b1


def train_step(weights, x, targets, learning_rate=0.05):
    probs, cache = forward(weights, x)
    backward(weights, cache, probs, targets, learning_rate)
    loss = -np.log(probs[np.arange(len(targets)), targets] + 1e-9).mean()
    return loss


def predict(weights, x):
    probs, _ = forward(weights, x)
    return probs.argmax(axis=-1)


def save_weights(weights, path):
    np.savez(path, **weights)


def load_weights(path):
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}
