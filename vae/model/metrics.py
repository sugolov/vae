import jax
import jax.random as random
import jax.numpy as jnp

import equinox as eqx
import equinox.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from vae.model.vqvae import VQVAE

def norm(x): return jnp.sqrt(jnp.sum((x)**2))
def sqdist(x1, x2): return jnp.sum((x1 - x2)**2)
def cossim(x1, x2): return jnp.sum(x1 * x2) / (norm(x1) * norm(x2))

def dot_kernel(X1, X2): return X2 @ X1.T

def sq_kernel(X1, X2):
    """||a - b||^2 = ||a||^2 - 2 * a @ b^T + ||b||^2"""
    a_sq = jnp.sum(X1 * X1, axis=1, keepdims=True).T
    b_sq = jnp.sum(X2 * X2, axis=1, keepdims=True)
    ab = X2 @ X1.T
    return - (a_sq - 2 * ab + b_sq)   


@jax.jit
def code_energy(codes, data, kernel=sq_kernel, beta=1):
    sims = kernel(codes, data)
    return -jax.nn.logsumexp(sims / beta, axis=0)

def code_grad(codes, data, kernel=sq_kernel, beta=1):
    sims = kernel(codes, data)
    smx = jax.nn.softmax(sims / beta, axis=0)
    return codes - smx.T @ data, -jax.nn.logsumexp(sims / beta, axis=0)

@jax.jit
def code_shift(codes, data, kernel=sq_kernel, beta=1):
    sims = kernel(codes, data)
    smx = jax.nn.softmax(sims / beta, axis=0)
    return smx.T @ data, -jax.nn.logsumexp(sims, axis=0)

def compute_code_energy(vqvae, batch):
    """
    computes the energy across all vqvae codes given the test batch
    """
    codes = vqvae.quantizer.codebook

    def _encode(x):
        enc = vqvae.encoder(x)
        enc = enc.reshape(-1, enc.shape[-1])
        return enc
    enc = jax.vmap(_encode)(batch)
    # flatten across all quantized encoding vectors
    enc = enc.reshape(-1, enc.shape[-1])

    return code_energy(codes, enc)



