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
    H = W = 16
    k = 2
    pad = 1

    layer_channels = [3, 8, 16, 64]

    x = jax.random.normal(key, (B, C, H, W))
    subkeys = jax.random.split(key, B)
    key = jax.random.PRNGKey(0)

    enc, state = nn.make_with_state(ConvAutoencoder)(layer_channels, key, kernel_size=k, padding=pad)
    print(enc)
    # print(eqx.nn.State(model=enc))

    x_out, z, state_out = jax.vmap(
        enc, 
        axis_name="batch", 
        in_axes=(0, None),
        out_axes=(0, None)
    )(x, state)

    print("AutoEnc x_out shape:", x_out.shape)
    print("AutoEnc z shape:", z.shape)

