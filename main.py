#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

import rasterio


class Model(nn.Module):
    n: int
    h: int

    def setup(self):
        self.layers = [nn.Dense(n) for n in [self.h] * self.n + [1]]

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

    args.outdir.mkdir(exist_ok=True)

    dataset = Dataset(args.input)
    sample = dataset.sample()

    targets = dataset.normalize(sample)
    targets = np.expand_dims(targets, axis=(0, -1))

    inputs = indices2d(targets.shape[1])
    inputs = np.expand_dims(inputs, axis=0)

    rng, seed = jax.random.split(rng)
    inputs = fourfeats(inputs, rng=seed, scale=args.ff_scale, size=2, features=args.ff_features)
    print(f"fourier features: scale={args.ff_scale}, features={args.ff_features}", file=sys.stderr)

    model = Model(n=args.nn_depth, h=args.nn_width)

    rng, seed = jax.random.split(rng)
    state = create_train_state(model, seed, inputs.shape, lr=args.nn_lr)

    num_params = sum(x.size for x in jax.tree_leaves(state.params))

    print(f"model: n={args.nn_depth}, h={args.nn_width}, params: {num_params}, lr={args.nn_lr}", file=sys.stderr)

    inputs = jax.device_put(inputs)
    targets = jax.device_put(targets)

    for step in range(1, args.train_steps + 1):
        grads, loss = apply_model(state, inputs, targets)
        state = update_model(state, grads)

        if step % args.train_save_freq == 0:
            outputs = state.apply_fn({"params": state.params}, inputs)
            outputs = np.squeeze(outputs)
            outputs = dataset.denormalize(outputs)

            errors = np.abs(outputs - sample)
            mean, std = errors.mean(), errors.std()
            emin, emax = errors.min(), errors.max()

            print(f"step: {step:5d}, loss: {loss:.5f}, mean: {mean:3.0f}m, std: {std:3.0f}m, min: {emin:3.0f}, max: {emax:3.0f}m", file=sys.stderr)

            with rasterio.open(args.input) as src:
                h, w = outputs.shape

                profile = src.profile
                profile.update(height=h, width=w)

                with rasterio.open(args.outdir / f"step-{step:05d}.tif", "w", **profile) as dst:
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


def create_train_state(model, rng, input_shape, lr):
    params = model.init(rng, jnp.zeros(input_shape))["params"]
    tx = optax.adam(lr)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", type=Path, help="Path to digital elevation model GeoTIFF")
    parser.add_argument("-o", "--outdir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("-w", "--nn-width", type=int, default=32, help="Number of features per layer")
    parser.add_argument("-d", "--nn-depth", type=int, default=8, help="Number of dense layers")
    parser.add_argument("-l", "--nn-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-f", "--ff-features", type=int, default=256, help="Number of fourier feature")
    parser.add_argument("-s", "--ff-scale", type=int, default=10, help="Fourier feature scale")
    parser.add_argument("-e", "--train-steps", type=int, default=1000, help="Steps to train for")
    parser.add_argument("-p", "--train-save-freq", type=int, default=10, help="Save every step")

    main(parser.parse_args())
