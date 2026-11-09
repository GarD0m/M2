from functools import partial

import jax
import jax.numpy as jnp


def rope(x, cos, sin, unsqueeze_dim=1):
    width = 2 * cos.shape[-1]
    if unsqueeze_dim is not None:
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    tail = x[..., -width:]
    pairs = tail.reshape(*tail.shape[:-1], width // 2, 2)
    other = jax.lax.rev(pairs, (pairs.ndim - 1,)).astype(jnp.float32)
    y = pairs.astype(jnp.float32) * cos[..., None] + other * (sin[..., None] * jnp.array([-1, 1], jnp.float32))
    return jnp.concatenate((x[..., :-width], y.reshape(tail.shape).astype(x.dtype)), axis=-1)


POLAR = (
    (8.1565545249, -22.4832929256, 15.8787699152),
    (4.0429299352, -2.8089174659, 0.5000178451),
    (3.8916678023, -2.7724841532, 0.5060648179),
    (3.2857536578, -2.3681294933, 0.4644902423),
    (2.3465413259, -1.7097828383, 0.4232355117),
)


def patch():
    from maxtext.layers import embeddings
    from maxtext.optimizers import optimizers
    from maxtext.utils import muon_utils

    embeddings._apply_rotary_pos_emb = rope
    muon_utils.ATTENTION_QKV_NAMES += ("qkv_proj",)
    optimizers.optax_muon = partial(muon, optimizers.optax_muon)


def muon(native, **kwargs):
    return native(**(kwargs | dict(ns_coeffs=POLAR, ns_steps=len(POLAR))))
