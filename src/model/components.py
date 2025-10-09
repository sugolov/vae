import os

import jax
import jax.numpy as jnp
from jax.nn import silu

import equinox as eqx
from equinox import nn

from jaxtyping import Array, PRNGKeyArray

class FeedForward(eqx.Module):
    proj1: nn.Linear
    proj2: nn.Linear

    def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray, *, 
                mlp_ratio=4.0):
        # intermediate = 4 * out_size
        k1, k2 = jax.random.split(key, 2)
        self.proj1 = nn.Linear(in_size, int(out_size*mlp_ratio), key=k1)
        self.proj2 = nn.Linear(int(out_size*mlp_ratio), out_size, key=k2)

    def __call__(self, x: Array, key: PRNGKeyArray):
        x = silu(self.proj1(x))
        x = silu(self.proj2(x))
        return x
        
class ConvPoolBlock(eqx.Module):
    conv: nn.Conv2d
    bn: nn.BatchNorm
    pool: nn.MaxPool2d

    def __init__(self, in_size: int, out_size: int, kernel_size: int, 
            padding: int, key: PRNGKeyArray):
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, 
                        stride=kernel_size, padding=padding, key=key
                    )
        self.bn = nn.BatchNorm(
            input_size=out_size, axis_name="batch"
        )
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=kernel_size)

    def __call__(self, x, state):
        x = self.conv(x)
        x, state = self.bn(x, state)
        x = silu(x)
        x = self.pool(x)
        return x, state