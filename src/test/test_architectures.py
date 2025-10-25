import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn

from model.architectures import ConvAutoencoder

def test_autoenc_forward():
    key = jax.random.PRNGKey(1)
    B = 4
    C = 3
    H = W = 32
    k = 2
    pad = 0
    stride = k

    layer_channels = [3, 16, 16, 64, 64]

    x = jax.random.normal(key, (B, C, H, W))
    subkeys = jax.random.split(key, B)
    key = jax.random.PRNGKey(0)

    enc, state = nn.make_with_state(ConvAutoencoder)(
        layer_channels, 
        key, 
        kernel_size=k, 
        padding=pad, 
        stride=stride
    )
    print(enc)
    

    x_out, z, state_out = jax.vmap(
        enc, 
        axis_name="batch", 
        in_axes=(0, None)
    )(x, state)

    print("AutoEnc x_out shape:", x_out.shape)
    print("AutoEnc z shape:", z.shape)