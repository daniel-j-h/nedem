#!/usr/bin/env python3

import sys
import argparse

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

import rasterio


class Model(nn.Module):
    def setup(self):
        self.layers = [nn.Dense(n) for n in [32] * 8 + [1]]

    def __call__(self, x):
        for i, lyr in enumerate(self.layers):
            x = lyr(x)

            if i != len(self.layers) - 1:
                x = nn.relu(x)

        return x


class Dataset:
    def __init__(self, path):
        with rasterio.open(path) as src:
            self.data = src.read(1)

        self.mean = self.data.mean()
        self.std = self.data.std()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return (x * self.std) + self.mean

    def sample(self):
        return self.data


def indices2d(n):
    xs = np.linspace(0, 1, n, endpoint=False)
    return np.stack(np.meshgrid(xs, xs), axis=-1)


# https://arxiv.org/abs/2006.10739
def fourfeats(inputs, *, rng, scale, size, features):
    bs = scale * jax.random.normal(rng, (features, size))
    inputs = (2. * jnp.pi * inputs) @ bs.T
    return jnp.concatenate([jnp.sin(inputs), jnp.cos(inputs)], axis=-1)


def main(args):
    rng = jax.random.PRNGKey(0)

    dataset = Dataset(args.input)
    sample = dataset.sample()

    targets = dataset.normalize(sample)
    targets = np.expand_dims(targets, axis=(0, -1))

    inputs = indices2d(targets.shape[1])
    inputs = np.expand_dims(inputs, axis=0)

    rng, seed = jax.random.split(rng)
    inputs = fourfeats(inputs, rng=seed, scale=10, size=2, features=256)

    rng, seed = jax.random.split(rng)
    state = create_train_state(seed, inputs.shape, lr=1e-4)

    num_params = sum(x.size for x in jax.tree_leaves(state.params))
    print(f"params: {num_params}", file=sys.stderr)

    inputs = jax.device_put(inputs)
    targets = jax.device_put(targets)

    for step in range(1, 2000 + 1):
        grads, loss = apply_model(state, inputs, targets)
        state = update_model(state, grads)

        if step % 10 == 0:
            outputs = state.apply_fn({"params": state.params}, inputs)
            outputs = np.squeeze(outputs)
            outputs = dataset.denormalize(outputs)

            errors = np.abs(outputs - sample)
            mean, std = errors.mean(), errors.std()
            emin, emax = errors.min(), errors.max()

            print(f"step: {step:3d}, loss: {loss:.5f}, mean: {mean:3.0f}m, std: {std:3.0f}m, min: {emin:3.0f}, max: {emax:3.0f}m", file=sys.stderr)

            with rasterio.open(args.input) as src:
                h, w = outputs.shape

                profile = src.profile
                profile.update(height=h, width=w)

                with rasterio.open(f"data/step-{step:05d}.tif", "w", **profile) as dst:
                    dst.write(outputs, 1)


@jax.jit
def apply_model(state, inputs, targets):
    def loss_fn(params):
        outputs = state.apply_fn({"params": params}, inputs)
        loss = jnp.mean(optax.l2_loss(outputs, targets))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    return grads, loss


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(rng, input_shape, lr):
    model = Model()
    params = model.init(rng, jnp.zeros(input_shape))["params"]
    tx = optax.adam(lr)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Path to digital elevation model GeoTIFF")

    main(parser.parse_args())
