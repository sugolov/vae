import os

import jax
import jax.numpy as jnp
from jax.nn import silu

import equinox as eqx
from equinox import nn

from typing import List, Optional, Any
from jaxtyping import Array, Key

from model.components import FeedForward, ConvBlock


class ConvAutoencoder(eqx.Module):
    layer_channels: List[int]
    latent_size: int
    num_blocks: int
    enc_blocks: List[ConvBlock]
    dec_blocks: List[ConvBlock]

    def __init__(self, layer_channels: List[int], key: Key, 
                 kernel_size: int = 2, stride: int = 2, 
                 padding: int = 1, use_logits: bool = False) -> None:
        """_summary_
        Args:
            layer_channels (List[int]): Layer channels in forward order for encoder
            key (Key): key
            kernel_size (int, optional): Convolutional kernel in each block. Defaults to 2.
            padding (int, optional): Conv layer padding. Defaults to 1.
            use_logits (bool, optional): Whether to view latent outputs as logits of a discrete. Defaults to False.
        """
        self.layer_channels = layer_channels
        self.latent_size = layer_channels[-1]
        self.num_blocks = 2 * (len(layer_channels) - 1)

        subkeys = jax.random.split(key, self.num_blocks)

        self.enc_blocks = []
        for i, (in_size, out_size) in enumerate(zip(layer_channels[:-1], layer_channels[1:])):
            self.enc_blocks.append(
                ConvBlock(in_size, 
                         out_size, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         stride=stride,
                         key=subkeys[i]
                )
            )
        
        # reverse the dims on this beast
        self.dec_blocks = []
        for j, (in_size, out_size) in enumerate(zip(layer_channels[1:][::-1], layer_channels[:-1][::-1])):
            self.dec_blocks.append(
                ConvBlock(in_size, 
                         out_size, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         stride=kernel_size,
                         key=subkeys[self.num_blocks + j],
                         transpose=True
                )
            )

    def __call__(self, x, state) -> Any:
        print(x.shape)
        for enc in self.enc_blocks:
            x, state = enc(x, state)
            print(x.shape)
        
        z = jnp.copy(x)
        
        for dec in self.dec_blocks:
            x, state = dec(x, state)
            print(x.shape)

        return x, z, state