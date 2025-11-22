import jax
import jax.numpy as jnp

import equinox as eqx
import equinox.nn as nn

import numpy as np
import matplotlib.pyplot as plt

class PCAProject:
    def __init__(self, x, n_components=2):
        self.n_components = n_components
        self.mean = np.mean(x, axis=0)
        
        # Center the data
        x_centered = x - self.mean
        U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
        
        self.components = Vt[:n_components]
        self.singular_values = S
        
    def __call__(self, x):
        x_centered = x - np.mean(x, axis=0)
        return x_centered @ self.components.T
    
from jaxtyping import Int, Float, Array

class LinearClassifier(eqx.Module):
    proj: eqx.nn.Linear | eqx.nn.Sequential

    def __init__(self, dim_in, num_classes, key=jax.random.PRNGKey(0), 
                 use_bias=True, use_softmax=True):
        layers = [nn.Linear(dim_in, num_classes, use_bias=use_bias, key=key)]
        if use_softmax:
            layers += [eqx.nn.Lambda(jax.nn.softmax)]
        self.proj = nn.Sequential(layers)

    def __call__(self, x):
        return self.proj(x)
    
@jax.jit
def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(jnp.log(pred_y), jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)

def classifier_loss(clf, x, y):
    pred_y = jax.vmap(clf)(x)
    return cross_entropy(y, pred_y)
