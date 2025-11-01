# test_encoder_decoder_shapes.py
import jax
import jax.numpy as jnp
from model.vqvae import Encoder, Decoder

def test_encoder_decoder_shapes():
    key = jax.random.PRNGKey(0)
    key_e, key_d, key_x = jax.random.split(key, 3)

    # config must match how Encoder/Decoder are constructed
    in_channels = 3
    ch = 128
    ch_mult = (1, 1, 2, 2, 4)
    num_res_blocks = 2
    z_channels = 256

    enc = Encoder(key_e, in_channels=in_channels, ch=ch, ch_mult=ch_mult,
                  num_res_blocks=num_res_blocks, z_channels=z_channels)
    dec = Decoder(key_d, out_channels=in_channels, ch=ch, ch_mult=ch_mult,
                  num_res_blocks=num_res_blocks, z_channels=z_channels)

    H, W = 32, 32
    x = jax.random.normal(key_x, (H, W, in_channels))

    z_e = enc(x)         # expected shape (Hq, Wq, z_channels)
    x_rec = dec(z_e)     # expected shape (H, W, C)

    print("input x shape:       ", x.shape)
    print("encoder output z_e:  ", z_e.shape)
    print("reconstructed x_rec: ", x_rec.shape)

    # compute expected spatial downsample factor from Encoder code:
    num_downsamples = len(ch_mult) - 1
    downsample_factor = 2 ** num_downsamples
    expected_Hq = H // downsample_factor
    expected_Wq = W // downsample_factor

    assert z_e.shape == (expected_Hq, expected_Wq, z_channels), \
        f"unexpected encoder output shape (got {z_e.shape}, expected {(expected_Hq, expected_Wq, z_channels)})"
    assert x_rec.shape == (H, W, in_channels), \
        f"unexpected decoder output shape (got {x_rec.shape}, expected {(H, W, in_channels)})"
