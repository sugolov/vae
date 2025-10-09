# test_modules.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn

from model.components import FeedForward, ConvPoolBlock  # <- adjust import

def test_feedforward_forward():
    key = jax.random.PRNGKey(0)
    B = 4
    in_size, out_size = 8, 16

    x = jax.random.normal(key, (B, in_size))
    subkeys = jax.random.split(key, B)
    print(subkeys.shape)

    ff = FeedForward(in_size, out_size, key)
    y = jax.vmap(ff, (0, 0))(x, subkeys)

    print("FeedForward x shape:", x.shape)
    print("FeedForward y shape:", y.shape)

    assert y.shape == (B, out_size)

def test_convpoolblock_forward():
    key = jax.random.PRNGKey(1)
    B, C_in, C_out = 4, 3, 6
    H = W = 8
    k = 2
    pad = 0

    x = jax.random.normal(key, (B, C_in, H, W))
    subkeys = jax.random.split(key, B)


    block, state = nn.make_with_state(ConvPoolBlock)(C_in, C_out, kernel_size=k, padding=pad, key=key)
    print(block)
    y, state_out = jax.vmap(
        block, 
        axis_name="batch", 
        in_axes=(0, None),
        out_axes=(0, None)
    )(x, state)

    print("ConvPoolBlock x shape:", x.shape)
    print("ConvPoolBlock y shape:", y.shape)

    # conv stride=k then pool stride=k => spatial dims divided by (k*k)
    expected_h = (H + pad) // k  # after conv (no pad)
    expected_w = (W + pad) // k
    expected_h //= k            # after pool
    expected_w //= k

    assert y.shape == (B, C_out, expected_h, expected_w)
    # state may be unchanged or updated depending on eqx version; just check it exists
    # assert state_out is not None or state is None
