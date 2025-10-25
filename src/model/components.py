import os

import jax
import jax.numpy as jnp
from jax.nn import silu

import equinox as eqx
from equinox import nn

from jaxtyping import Array, Key

from utils.conv import conv_shape

class FeedForward(eqx.Module):
    proj1: nn.Linear
    proj2: nn.Linear

    def __init__(self, in_size: int, out_size: int, key: Key, *, 
                mlp_ratio=4.0):
        # intermediate = 4 * out_size
        k1, k2 = jax.random.split(key, 2)
        self.proj1 = nn.Linear(in_size, int(out_size*mlp_ratio), key=k1)
        self.proj2 = nn.Linear(int(out_size*mlp_ratio), out_size, key=k2)

    def __call__(self, x: Array, key: Key):
        x = silu(self.proj1(x))
        x = silu(self.proj2(x))
        return x
        
class ConvBlock(eqx.Module):
    conv: nn.Conv2d | nn.ConvTranspose2d
    norm: nn.BatchNorm 
    pool: nn.MaxPool2d

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 padding: int, stride: int | None, img_shape: tuple[int] | None = None,
                 transpose: bool = False, norm: str = "batchnorm", 
                 key: Key = jax.random.PRNGKey(0)):
        
        stride = kernel_size if not stride else stride

        Conv = nn.Conv2d if not transpose else nn.ConvTranspose2d

        self.conv = Conv(in_channels, 
                         out_channels, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         padding=padding, 
                         key=key
                    )
        
        if norm == "layernorm":
            assert img_shape is not None
            # TODO: implement with correct shapes
            norm_shape = down_conv_shape(
                **img_shape, 
                stride=stride,
                kernel=kernel_size, 
                pad=pad
            )
            self.norm = nn.LayerNorm(
                conv_shape(**img_shape, stride=stride)
            )
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm(
                input_size=out_channels, axis_name="batch"
            )

        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=0, stride=kernel_size)

    def __call__(self, x, state):
        x = self.conv(x)
        x, state = self.norm(x, state)
        x = silu(x)
        return x, state
    

class ResBlock(eqx.Module):
    norm_fn1: eqx.nn.GroupNorm
    norm_fn2: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(self, input_dim, filters, key):
        key1, key2, key3 = jax.random.split(key, 3)

        num_groups1 = min(32, input_dim) if input_dim >= 32 else max(1, input_dim // 4)
        num_groups2 = min(32, filters) if filters >= 32 else max(1, filters // 4)

        self.norm_fn1 = eqx.nn.GroupNorm(num_groups1, input_dim)
        self.norm_fn2 = eqx.nn.GroupNorm(num_groups2, filters)

        self.conv1 = eqx.nn.Conv2d(input_dim, filters, (3, 3), padding=1, use_bias=False, key=key1)
        self.conv2 = eqx.nn.Conv2d(filters, filters, (3, 3), padding=1, use_bias=False, key=key2)

        if input_dim != filters:
            self.conv3 = eqx.nn.Conv2d(input_dim, filters, (1, 1), use_bias=False, key=key3)
        else:
            self.conv3 = None

    def __call__(self, x):
        residual = x
        x = self.norm_fn1(x)
        x = jax.nn.swish(x)
        x = self.conv1(x)
        x = self.norm_fn2(x)
        x = jax.nn.swish(x)
        x = self.conv2(x)

        if self.conv3 is not None:
            residual = self.conv3(residual)
        return x + residual