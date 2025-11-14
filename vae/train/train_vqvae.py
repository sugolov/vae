import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from tqdm import tqdm
import argparse
import aim
import os
import functools
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import pickle
import requests
import tarfile
from typing import Tuple

from vae.data import build_dataset
from vae.model.vqvae import VQVAE, train_step
from vae.train.fid import compute_frechet_distance, compute_statistics

from vae.train.flax_inception import InceptionV3

@eqx.filter_jit
def batch_reconstruct(vqvae, imgs_batch):
    """JIT-compiled batch reconstruction"""
    def single_reconstruct(img):
        outputs = vqvae.forward(img)
        return outputs["reconstruction"]
    return jax.vmap(single_reconstruct)(imgs_batch)


def compute_fid_score(vqvae, train_images, mu_real, sigma_real, inception_params, apply_fn, n_samples=10000, batch_size=256):
    n_processed = 0
    all_recons = []

    # Sample random images for FID computation
    indices = np.random.permutation(len(train_images))[:n_samples]
    n_batches_needed = (n_samples + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches_needed), desc="Computing FID", leave=False):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        imgs_batch = train_images[batch_indices]
        imgs_jax = jnp.array(imgs_batch)

        recons = batch_reconstruct(vqvae, imgs_jax)

        recons_np = np.array(recons)
        recons_np = np.clip(recons_np, 0, 1)
        recons_np = (recons_np * 255).astype(np.uint8)

        all_recons.append(recons_np)

    all_recons = np.concatenate(all_recons, axis=0)

    print(f"Computing InceptionV3 features for {len(all_recons)} reconstructions...")
    mu_recon, sigma_recon = compute_statistics(
        all_recons, inception_params, apply_fn, batch_size=256, img_size=(256, 256)
    )

    fid = compute_frechet_distance(mu_real, mu_recon, sigma_real, sigma_recon, eps=1e-6)
    return fid


def save_reconstruction_grid(vqvae, train_images, epoch, save_path, n_images=25):
    n_rows = int(np.sqrt(n_images))
    n_cols = (n_images + n_rows - 1) // n_rows

    # Sample random images for visualization
    indices = np.random.permutation(len(train_images))[:n_images]
    imgs_batch = train_images[indices]
    imgs_jax = jnp.array(imgs_batch)

    recons = []
    for i in range(min(n_images, imgs_jax.shape[0])):
        outputs = vqvae.forward(imgs_jax[i])
        recons.append(outputs["reconstruction"])
    recons = jnp.stack(recons)

    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = gridspec.GridSpec(n_rows * 2, n_cols, hspace=0.3)

    for i in range(n_images):
        row = (i // n_cols) * 2
        col = i % n_cols

        ax1 = fig.add_subplot(gs[row, col])
        img = np.array(imgs_jax[i])
        ax1.imshow(np.clip(img, 0, 1))
        ax1.axis('off')
        if i < n_cols:
            ax1.set_title('Original', fontsize=10)

        ax2 = fig.add_subplot(gs[row + 1, col])
        recon = np.array(recons[i])
        ax2.imshow(np.clip(recon, 0, 1))
        ax2.axis('off')
        if i < n_cols:
            ax2.set_title('Reconstruction', fontsize=10)

    plt.suptitle(f'Epoch {epoch}', fontsize=14)
    plt.savefig(f"{save_path}_epoch_{epoch}.png", dpi=100, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, opt_state, epoch, args, path):
    checkpoint = {
        'epoch': epoch,
        'args': vars(args)
    }
    eqx.tree_serialise_leaves(path + "_model.eqx", model)
    eqx.tree_serialise_leaves(path + "_opt.eqx", opt_state)

    import json
    with open(path + "_meta.json", "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint(path, seed):
    import json
    with open(path + "_meta.json", "r") as f:
        checkpoint = json.load(f)

    args = argparse.Namespace(**checkpoint['args'])
    ch_mult = tuple(map(int, args.ch_mult.split(',')))

    key = jax.random.key(seed)
    model = VQVAE(
        key=key,
        in_channels=3,
        ch=args.ch,
        ch_mult=ch_mult,
        num_res_blocks=args.num_res_blocks,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        beta_commit=args.beta_commit,
        ema_decay=0.95,  # Lower decay for CIFAR-10
        epsilon=1e-5
    )

    model = eqx.tree_deserialise_leaves(path + "_model.eqx", model)

    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = eqx.tree_deserialise_leaves(path + "_opt.eqx", opt_state)

    return model, opt_state, checkpoint['epoch'], args

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="vqvae_cifar10")
    p.add_argument("--tag", type=str, default=None)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--vis_interval", type=int, default=10)

    p.add_argument("--n_fid_samples", type=int, default=10_000)
    p.add_argument("--no_fid", action="store_true", help="Skip FID computation for testing")

    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_name", type=str, default="CIFAR10")
    p.add_argument("--data_dir", type=str, default="./data/")
    p.add_argument("--aim_repo", type=str, default=None, help="Path to aim repository (defaults to .aim in current dir)")
    
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--ch", type=int, default=128)
    p.add_argument("--ch_mult", type=str, default="1,2,4")
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--num_embeddings", type=int, default=1024)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--beta_commit", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def train_cifar10(args):
    print(f"JAX backend: {jax.devices()[0].platform}")
    print(f"JAX devices: {jax.devices()}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if args.tag is None:
        tag = tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        tag = args.tag
    run_name = "_".join([args.exp_name, tag])

    ch_mult = tuple(map(int, args.ch_mult.split(',')))
    key = jax.random.key(args.seed)

    # create dataset
    print("Loading CIFAR-10 dataset...")
    dataloader, num_classes, n_train, image_size = build_dataset(
        args.data_name,
        args.data_dir,
        batch_size=args.batch_size,
        is_train=True,
        num_workers=4,
    )

    n_batches = n_train // args.batch_size
    print(f"Loaded {n_train} training images")
    
    # Store first batch for visualization/testing - convert to HWC
    first_batch_imgs, _ = next(iter(dataloader))
    test_batch = first_batch_imgs[:100]  # Keep 100 images for visualization
    # test_batch = np.transpose(test_batch, (0, 2, 3, 1))  # BCHW -> BHWC
    print(f"Test batch shape: {test_batch.shape}")
    
    # resume from checkpoint
    if args.resume:
        vqvae, opt_state, start_epoch, args = load_checkpoint(args.resume, args.seed)
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        key, model_key = jax.random.split(key)
        vqvae = VQVAE(
            key=model_key,
            in_channels=3,
            ch=args.ch,
            ch_mult=ch_mult,
            num_res_blocks=args.num_res_blocks,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            beta_commit=args.beta_commit,
            ema_decay=0.95,
            epsilon=1e-5
        )

        schedule = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=args.epochs * n_batches
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(eqx.filter(vqvae, eqx.is_array))

    n_params = sum(x.size for x in jax.tree.leaves(vqvae))
    print(f"Model parameters: {n_params:,}")

    # Check latent shape - encoder expects single image without batch dim in HWC
    test_img = test_batch[0]  # Shape: (32, 32, 3)
    print(f"Test image shape: {test_img.shape}")
    test_latent = vqvae.encoder(test_img)
    print(f"Latent shape for {image_size}x{image_size} input: {test_latent.shape} (expecting 8x8x{args.embedding_dim} with --ch_mult 1,2,4)")

    # Initialize InceptionV3 for FID computation (only if needed)
    if not args.no_fid:
        print("Initializing InceptionV3 for FID...")
        rng = jax.random.key(0)
        inception = InceptionV3(pretrained=True)
        inception_params = inception.init(rng, jnp.ones((1, 32, 32, 3)))
        apply_fn = jax.jit(functools.partial(inception.apply, train=False))

        # Precompute or load real CIFAR-10 statistics
        stats_path = Path(args.save_dir) / "fid_stats_cifar10.npz"
        if stats_path.exists():
            with np.load(stats_path) as f:
                mu_real, sigma_real = f["mu"], f["sigma"]
            print(f"Loaded FID stats from {stats_path}")
        else:
            print("Computing FID statistics for real CIFAR-10 data...")

            real_images_list = []
            for imgs_batch, _ in dataloader:
                real_images_list.append(imgs_batch)
                if len(real_images_list) * args.batch_size >= 10000:  # Use 10k images for stats
                    break
            real_images = np.concatenate(real_images_list, axis=0)[:10000]
            real_images_uint8 = (real_images * 255).astype(np.uint8)

            mu_real, sigma_real = compute_statistics(
                real_images_uint8, inception_params, apply_fn, batch_size=256, img_size=(256, 256)
            )

            np.savez(stats_path, mu=mu_real, sigma=sigma_real)
            print(f"Saved FID stats to {stats_path}")
    else:
        print("Skipping FID initialization (--no_fid flag set)")
        mu_real = sigma_real = inception_params = apply_fn = None

    logf = open(f"{args.save_dir}/{run_name}_log.txt", "a" if args.resume else "w")
    if not args.resume:
        if args.no_fid:
            logf.write("Epoch,Train_Loss,Recon_Loss,Commit_Loss\n")
        else:
            logf.write("Epoch,Train_Loss,Recon_Loss,Commit_Loss,FID\n")

    @eqx.filter_jit
    def jit_train_step(model, batch, opt_state, key):
        return train_step(model, batch, opt_state, optimizer.update, key)

    if start_epoch == 0:
        vis_path = os.path.join(args.save_dir, f"{run_name}_reconstructions")
        save_reconstruction_grid(vqvae, test_batch, 0, vis_path, n_images=25)
        print(f"Saved initial reconstruction grid to {vis_path}_epoch_0.png")

        if not args.no_fid:
            print("Computing initial FID score...")
            # For FID, collect sample images
            sample_images = []
            for imgs_batch, _ in dataloader:
                # Convert BCHW to BHWC
                # imgs_batch = np.transpose(imgs_batch, (0, 2, 3, 1))
                sample_images.append(imgs_batch)
                if len(sample_images) * args.batch_size >= args.n_fid_samples:
                    break
            sample_images = np.concatenate(sample_images, axis=0)[:args.n_fid_samples]
            
            fid_score = compute_fid_score(vqvae, sample_images, mu_real, sigma_real, inception_params, apply_fn, args.n_fid_samples, args.batch_size)
            logf.write(f"0,0.0000,0.0000,0.0000,{fid_score:.2f}\n")
            logf.flush()
            print(f"Epoch 0 (untrained): FID={fid_score:.2f}")
        else:
            logf.write(f"0,0.0000,0.0000,0.0000\n")
            logf.flush()
            print(f"Epoch 0 (untrained): FID=skipped")

    # Rolling buffer of recent latents for refresh
    buffer_max = 8192
    z_buffer = np.empty((0, args.embedding_dim), dtype=np.float32)


    # aim logging
    run = aim.Run(repo=args.aim_repo, experiment=args.exp_name)
    run["hparams"] = vars(args)

    # train loop
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = {"total": 0, "recon": 0, "commit": 0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for imgs_batch, labels_batch in pbar:
            # Convert BCHW to BHWC for JAX model
            # imgs_batch = np.transpose(imgs_batch, (0, 2, 3, 1))
            imgs_jax = jnp.array(imgs_batch)

            key, subkey = jax.random.split(key)
            vqvae, opt_state, loss, outputs = jit_train_step(vqvae, imgs_jax, opt_state, subkey)

            batch_size_actual = imgs_jax.shape[0]
            recon_losses = outputs["recon_loss"]
            commit_losses = outputs["commit_loss"]

            epoch_losses["total"] += float(loss) * batch_size_actual
            epoch_losses["recon"] += float(jnp.sum(recon_losses))
            epoch_losses["commit"] += float(jnp.sum(commit_losses))

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "recon": f"{jnp.mean(recon_losses):.6f}",
                "commit": f"{jnp.mean(commit_losses):.8f}"
            })

            # Collect z_e into a host-side buffer
            z_batch = np.array(outputs["z_e"]).reshape(-1, args.embedding_dim)
            if z_buffer.size == 0:
                z_buffer = z_batch
            else:
                z_buffer = np.concatenate([z_buffer, z_batch], axis=0)
            if z_buffer.shape[0] > buffer_max:
                z_buffer = z_buffer[-buffer_max:]

        avg_losses = {k: v / n_train for k, v in epoch_losses.items()}
        # track with aim
        _ = [run.track(v, name=k) for k, v in avg_losses.items()]

        # Compute FID only at log intervals and if not disabled
        if (epoch + 1) % args.log_interval == 0 and not args.no_fid:
            print(f"\nEpoch {epoch+1}: Computing FID score...")
            # Collect sample images for FID
            sample_images = []
            for imgs_batch, _ in dataloader:
                # Convert BCHW to BHWC
                # imgs_batch = np.transpose(imgs_batch, (0, 2, 3, 1))
                sample_images.append(imgs_batch)
                if len(sample_images) * args.batch_size >= args.n_fid_samples:
                    break
            sample_images = np.concatenate(sample_images, axis=0)[:args.n_fid_samples]
            
            fid_score = compute_fid_score(vqvae, sample_images, mu_real, sigma_real, inception_params, apply_fn, args.n_fid_samples, args.batch_size)

            print(f"Epoch {epoch+1}: Loss={avg_losses['total']:.4f}, "
                  f"Recon={avg_losses['recon']:.6f}, Commit={avg_losses['commit']:.8f}, "
                  f"FID={fid_score:.2f}")
        else:
            fid_score = None
            if args.no_fid:
                print(f"Epoch {epoch+1}: Loss={avg_losses['total']:.4f}, "
                      f"Recon={avg_losses['recon']:.6f}, Commit={avg_losses['commit']:.8f}")
            else:
                print(f"Epoch {epoch+1}: Loss={avg_losses['total']:.4f}, "
                      f"Recon={avg_losses['recon']:.6f}, Commit={avg_losses['commit']:.8f}, "
                      f"FID=pending")

        # Write losses every epoch
        log_str = f"{epoch+1},{avg_losses['total']:.6f},{avg_losses['recon']:.8f},{avg_losses['commit']:.6f}"
        if not args.no_fid:
            if fid_score is not None and isinstance(fid_score, float):
                log_str += f",{fid_score:.2f}"
            else:
                log_str += ",N/A"
        logf.write(log_str + "\n")
        logf.flush()

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"{run_name}_epoch_{epoch+1}")
            save_checkpoint(vqvae, opt_state, epoch + 1, args, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if (epoch + 1) % args.vis_interval == 0:
            vis_path = os.path.join(args.save_dir, f"{run_name}_reconstructions")
            save_reconstruction_grid(vqvae, test_batch, epoch + 1, vis_path, n_images=25)
            print(f"Saved reconstruction grid to {vis_path}_epoch_{epoch+1}.png")

    final_path = os.path.join(args.save_dir, f"{run_name}_final")
    save_checkpoint(vqvae, opt_state, args.epochs, args, final_path)
    print(f"Training complete! Final model saved to {final_path}")

    vis_path = os.path.join(args.save_dir, f"{run_name}_reconstructions")
    save_reconstruction_grid(vqvae, test_batch, args.epochs, vis_path, n_images=25)
    print(f"Saved final reconstruction grid to {vis_path}_epoch_{args.epochs}.png")

    logf.close()

if __name__ == "__main__":
    args = parse_args()
    train_cifar10(args)