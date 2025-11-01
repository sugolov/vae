import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array, Int

def squared_euclidean_distance(
    a: Float[Array, "N D"],
    b: Float[Array, "K D"]
) -> Float[Array, "N K"]:
    """||a - b||^2 = ||a||^2 - 2 * a @ b^T + ||b||^2"""
    a_sq = jnp.sum(a * a, axis=1, keepdims=True)
    b_sq = jnp.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    return a_sq - 2 * ab + b_sq


def flatten_spatial(x: Float[Array, "H W D"]) -> Float[Array, "HW D"]:
    H, W, D = x.shape
    return x.reshape(H * W, D)


def unflatten_spatial(
    x: Float[Array, "HW D"],
    H: int,
    W: int
) -> Float[Array, "H W D"]:
    D = x.shape[-1]
    return x.reshape(H, W, D)


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


class Encoder(eqx.Module):
    conv_in: eqx.nn.Conv2d
    down_blocks: list
    mid_block1: ResBlock
    mid_block2: ResBlock
    norm_out: eqx.nn.GroupNorm
    conv_out: eqx.nn.Conv2d

    def __init__(
        self,
        key,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        z_channels=256,
    ):
        num_keys = 1 + len(ch_mult) * num_res_blocks + (len(ch_mult) - 1) + 3
        keys = jax.random.split(key, num_keys)
        key_idx = 0

        self.conv_in = eqx.nn.Conv2d(in_channels, ch, (3, 3), padding=1, key=keys[key_idx])
        key_idx += 1

        self.down_blocks = []
        curr_ch = ch
        for i, mult in enumerate(ch_mult):
            block = []
            ch_out = ch * mult
            for j in range(num_res_blocks):
                block.append(ResBlock(curr_ch, ch_out, keys[key_idx]))
                key_idx += 1
                curr_ch = ch_out
            if i != len(ch_mult) - 1:
                block.append(eqx.nn.Conv2d(curr_ch, curr_ch, (3, 3), stride=2, padding=1, key=keys[key_idx]))
                key_idx += 1
            self.down_blocks.append(block)

        self.mid_block1 = ResBlock(curr_ch, curr_ch, keys[key_idx])
        key_idx += 1
        self.mid_block2 = ResBlock(curr_ch, curr_ch, keys[key_idx])
        key_idx += 1

        num_groups = min(32, curr_ch) if curr_ch >= 32 else max(1, curr_ch // 4)
        self.norm_out = eqx.nn.GroupNorm(num_groups, curr_ch)
        self.conv_out = eqx.nn.Conv2d(curr_ch, z_channels, (3, 3), padding=1, key=keys[key_idx])

    def __call__(self, x: Float[Array, "H W C"]) -> Float[Array, "Hq Wq D"]:
        x = jnp.transpose(x, (2, 0, 1))

        x = self.conv_in(x)

        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
                print(x.shape)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        x = self.norm_out(x)
        x = jax.nn.swish(x)
        x = self.conv_out(x)

        x = jnp.transpose(x, (1, 2, 0))
        return x


class Decoder(eqx.Module):
    conv_in: eqx.nn.Conv2d
    mid_block1: ResBlock
    mid_block2: ResBlock
    up_blocks: list
    norm_out: eqx.nn.GroupNorm
    conv_out: eqx.nn.Conv2d

    def __init__(
        self,
        key,
        out_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        z_channels=256,
    ):
        ch_mult = tuple(reversed(ch_mult))

        num_keys = 1 + 2 + len(ch_mult) * num_res_blocks + (len(ch_mult) - 1) + 2
        keys = jax.random.split(key, num_keys)
        key_idx = 0

        block_in = ch * ch_mult[0]
        self.conv_in = eqx.nn.Conv2d(z_channels, block_in, (3, 3), padding=1, key=keys[key_idx])
        key_idx += 1

        self.mid_block1 = ResBlock(block_in, block_in, keys[key_idx])
        key_idx += 1
        self.mid_block2 = ResBlock(block_in, block_in, keys[key_idx])
        key_idx += 1

        self.up_blocks = []
        curr_ch = block_in
        for i, mult in enumerate(ch_mult):
            block = []
            ch_out = ch * mult
            for j in range(num_res_blocks):
                block.append(ResBlock(curr_ch, ch_out, keys[key_idx]))
                key_idx += 1
                curr_ch = ch_out
            if i != len(ch_mult) - 1:
                block.append(eqx.nn.ConvTranspose2d(curr_ch, curr_ch, (4, 4), stride=2, padding=1, key=keys[key_idx]))
                key_idx += 1
            self.up_blocks.append(block)

        num_groups = min(32, curr_ch) if curr_ch >= 32 else max(1, curr_ch // 4)
        self.norm_out = eqx.nn.GroupNorm(num_groups, curr_ch)
        self.conv_out = eqx.nn.Conv2d(curr_ch, out_channels, (3, 3), padding=1, key=keys[key_idx])

    def __call__(self, z: Float[Array, "Hq Wq D"]) -> Float[Array, "H W C"]:
        x = jnp.transpose(z, (2, 0, 1))


        x = self.conv_in(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
                print(layer, x.shape)

        x = self.norm_out(x)
        x = jax.nn.swish(x)
        x = self.conv_out(x)

        x = jnp.transpose(x, (1, 2, 0))
        return x

class VectorQuantizer(eqx.Module):
    codebook: Float[Array, "K D"]
    ema_cluster_size: Float[Array, "K"]
    ema_embedding_sum: Float[Array, "K D"]

    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)
    ema_decay: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings = 512,
        embedding_dim = 64,
        ema_decay = 0.99,
        epsilon = 1e-5,
        key = None
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.codebook = jax.random.normal(
            key, (num_embeddings, embedding_dim)
        ) * 0.02
        self.ema_cluster_size = jnp.zeros(num_embeddings)
        self.ema_embedding_sum = jnp.zeros((num_embeddings, embedding_dim))

    def quantize(
        self,
        z_e: Float[Array, "H W D"]
    ) -> tuple[Float[Array, "H W D"], Int[Array, "H W"], Float[Array, "HW"]]:
        H, W, D = z_e.shape
        z_e_flat = flatten_spatial(z_e)
        distances = squared_euclidean_distance(z_e_flat, self.codebook)
        indices_flat = jnp.argmin(distances, axis=1)
        min_distances = jnp.min(distances, axis=1)
        z_q_flat = self.codebook[indices_flat]
        z_q = unflatten_spatial(z_q_flat, H, W)
        indices = indices_flat.reshape(H, W)
        return z_q, indices, min_distances

    def straight_through(
        self,
        z_e: Float[Array, "H W D"],
        z_q: Float[Array, "H W D"]
    ) -> Float[Array, "H W D"]:
        return z_e + jax.lax.stop_gradient(z_q - z_e)

    def compute_ema_statistics(
        self,
        z_e: Float[Array, "H W D"],
        indices: Int[Array, "H W"]
    ) -> tuple[Float[Array, "K"], Float[Array, "K D"]]:
        H, W, D = z_e.shape
        z_e_flat = flatten_spatial(z_e)
        indices_flat = indices.reshape(-1)
        batch_cluster_size = jnp.bincount(indices_flat, length=self.num_embeddings).astype(jnp.float32)
        batch_embedding_sum = jnp.zeros((self.num_embeddings, D))
        batch_embedding_sum = batch_embedding_sum.at[indices_flat].add(z_e_flat)

        return batch_cluster_size, batch_embedding_sum

    def update_codebook_ema(
        self,
        batch_cluster_size: Float[Array, "K"],
        batch_embedding_sum: Float[Array, "K D"],
        z_e_flat: Float[Array, "N D"] = None,
        key = None
    ):
        new_ema_cluster_size = (
            self.ema_decay * self.ema_cluster_size +
            (1 - self.ema_decay) * batch_cluster_size
        )
        new_ema_embedding_sum = (
            self.ema_decay * self.ema_embedding_sum +
            (1 - self.ema_decay) * batch_embedding_sum
        )
        n = new_ema_cluster_size[:, None] + self.epsilon
        new_codebook = new_ema_embedding_sum / n

        # if z_e_flat is not None and key is not None:
        #     usage = new_ema_cluster_size > 1.0
        #     n_unused = jnp.sum(~usage)
        #     if n_unused > 0:
        #         n_samples = z_e_flat.shape[0]
        #         random_indices = jax.random.randint(
        #             key, shape=(self.num_embeddings,),
        #             minval=0, maxval=n_samples
        #         )
        #         random_samples = z_e_flat[random_indices]
        #         new_codebook = jnp.where(usage[:, None], new_codebook, random_samples)

        return eqx.tree_at(
            lambda q: (q.codebook, q.ema_cluster_size, q.ema_embedding_sum),
            self,
            (new_codebook, new_ema_cluster_size, new_ema_embedding_sum)
        )

class VQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: VectorQuantizer
    beta_commit: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        embedding_dim=256,
        num_embeddings=512,
        beta_commit=0.25,
        ema_decay=0.99,
        epsilon=1e-5
    ):
        key_e, key_d, key_q = jax.random.split(key, 3)

        self.encoder = Encoder(
            key_e,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embedding_dim
        )

        self.decoder = Decoder(
            key_d,
            out_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=embedding_dim
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            ema_decay=ema_decay,
            epsilon=epsilon,
            key=key_q
        )

        self.beta_commit = beta_commit

    def encode(self, x: Float[Array, "H W C"]) -> Int[Array, "Hq Wq"]:
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer.quantize(z_e)
        return indices

    def decode(self, indices: Int[Array, "Hq Wq"]) -> Float[Array, "H W C"]:
        z_q_flat = self.quantizer.codebook[indices.reshape(-1)]
        H, W = indices.shape
        z_q = unflatten_spatial(z_q_flat, H, W)
        return self.decoder(z_q)

    def forward(self, x: Float[Array, "H W C"]) -> dict:
        z_e = self.encoder(x)
        z_q, indices, distances = self.quantizer.quantize(z_e)
        z_q_st = self.quantizer.straight_through(z_e, z_q)
        x_recon = self.decoder(z_q_st)

        recon_loss = jnp.mean((x - x_recon) ** 2)
        vq_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
        commit_loss = self.beta_commit * jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

        return {
            "reconstruction": x_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
            "commit_loss": commit_loss,
            "distances": distances
        }


def _single_vqvae_loss(vqvae, data):
    outputs = vqvae.forward(data)
    # With EMA updates, we don't backprop through vq_loss
    total_loss = outputs["recon_loss"] + outputs["commit_loss"]
    return total_loss, outputs

def vqvae_loss(vqvae, data):
    losses, outputs = eqx.filter_vmap(_single_vqvae_loss, in_axes=(None, 0))(vqvae, data)
    return jnp.mean(losses), outputs

def train_step(vqvae, data, opt_state, opt_update, key):
    def loss_fn(model, x):
        losses, outputs = vqvae_loss(model, x)
        return losses, outputs

    (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(vqvae, data)

    grads = eqx.tree_at(
        lambda g: g.quantizer.codebook,
        grads,
        jnp.zeros_like(vqvae.quantizer.codebook)
    )

    updates, opt_state = opt_update(grads, opt_state)
    vqvae = eqx.apply_updates(vqvae, updates)

    batch_cluster_size, batch_embedding_sum = jax.vmap(
        lambda out: vqvae.quantizer.compute_ema_statistics(out["z_e"], out["indices"])
    )(outputs)

    batch_cluster_size = jnp.sum(batch_cluster_size, axis=0)
    batch_embedding_sum = jnp.sum(batch_embedding_sum, axis=0)

    all_z_e = outputs["z_e"].reshape(-1, outputs["z_e"].shape[-1])

    new_quantizer = vqvae.quantizer.update_codebook_ema(
        batch_cluster_size, batch_embedding_sum, all_z_e, key
    )
    vqvae = eqx.tree_at(lambda m: m.quantizer, vqvae, new_quantizer)

    return vqvae, opt_state, loss, outputs