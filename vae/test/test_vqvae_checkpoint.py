import os
import tempfile
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from vae.model.vqvae import VQVAE
from vae.train.train_vqvae import save_checkpoint, load_checkpoint


def _args_for_test():
    return argparse.Namespace(
        exp_name="ckpt_test",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        log_interval=1,
        save_interval=1,
        vis_interval=1,
        n_fid_samples=128,
        no_fid=True,
        save_dir="./_tmp_ckpts",
        data_name="CIFAR10",
        data_dir="./data",
        resume=None,
        ch=32,
        ch_mult="1,2,4",
        num_res_blocks=1,
        num_embeddings=64,
        embedding_dim=16,
        beta_commit=0.25,
        seed=123,
    )


def _make_test_model_and_state(args):
    ch_mult = tuple(map(int, args.ch_mult.split(",")))
    key = jax.random.key(args.seed)
    key, model_key = jax.random.split(key)

    model = VQVAE(
        key=model_key,
        in_channels=3,
        ch=args.ch,
        ch_mult=ch_mult,
        num_res_blocks=args.num_res_blocks,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        beta_commit=args.beta_commit,
        ema_decay=0.95,
        epsilon=1e-5,
    )
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return model, opt_state


def test_save_checkpoint():
    """Ensure save_checkpoint writes all expected files."""
    args = _args_for_test()
    model, opt_state = _make_test_model_and_state(args)

    path = os.path.join(tempfile.gettempdir(), "vqvae_ckpt_test_save")
    save_checkpoint(model, opt_state, epoch=3, args=args, path=path)

    for suffix in ["_model.eqx", "_opt.eqx", "_meta.json"]:
        fpath = path + suffix
        assert os.path.exists(fpath), f"Missing checkpoint file: {fpath}"

    print("✅ save_checkpoint successfully wrote all files.")


def test_load_checkpoint():
    """Ensure load_checkpoint correctly restores model, optimizer, and args."""
    args = _args_for_test()
    model0, opt_state0 = _make_test_model_and_state(args)
    base = os.path.join(tempfile.gettempdir(), "vqvae_ckpt_test_load")

    save_checkpoint(model0, opt_state0, epoch=7, args=args, path=base)
    model1, opt_state1, epoch1, args1 = load_checkpoint(base, seed=args.seed)

    # epoch round-trip
    assert epoch1 == 7

    # args round-trip
    assert vars(args) == vars(args1)

    # model weights identical
    leaves0 = jax.tree_util.tree_leaves(eqx.filter(model0, eqx.is_array))
    leaves1 = jax.tree_util.tree_leaves(eqx.filter(model1, eqx.is_array))
    for x, y in zip(leaves0, leaves1):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=0, atol=0)

    # optimizer state identical
    opt0 = jax.tree_util.tree_leaves(opt_state0)
    opt1 = jax.tree_util.tree_leaves(opt_state1)
    for x, y in zip(opt0, opt1):
        np.testing.assert_allclose(np.array(x), np.array(y), rtol=0, atol=0)

    print("✅ load_checkpoint successfully restored model, optimizer, and args.")
