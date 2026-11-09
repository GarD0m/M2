import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name


def bits(x):
    x = jax.lax.bitcast_convert_type(x, jnp.int32)
    return x ^ ((x >> 31) & 0x7fffffff)


def top(scores, k):
    values, ids = jax.lax.top_k(scores, k)
    edge = bits(values[..., -1:])
    scores = bits(scores)
    return (scores > edge) | ((scores == edge) & (jnp.arange(scores.shape[-1]) <= ids[..., -1:]))


def patch():
    from maxtext.layers import attention_compressed as csa

    cls = csa.DeepseekV4CSACompressor
    native = cls.__call__

    def pool(m, x, pos, width):
        return csa.compute_csa_prefill_chunk_pooling(
            kv=m.kv_proj(x), gate=m.gate_proj(x), seq_len=x.shape[1], batch_size=x.shape[0],
            compress_rate=m.compress_rate, position_ids=pos, position_bias=m.position_bias.value,
            kv_norm=m.kv_norm, rotary_emb=m.rotary_emb, head_dim=width, dtype=m.dtype,
        )[0]

    def compress(self, hidden_states, q_latent, position_ids, attention_mask=None,
                 model_mode=csa.MODEL_MODE_TRAIN, cache=None, indexer_cache=None,
                 use_indexer=True, return_indexer_scores=False):
        x, pos = hidden_states, position_ids
        if model_mode != csa.MODEL_MODE_TRAIN or cache is not None or indexer_cache is not None or not use_indexer:
            return native(self, x, q_latent, pos, attention_mask, model_mode, cache, indexer_cache,
                          use_indexer, return_indexer_scores)

        idx = self.indexer
        batch, length, _ = x.shape
        inp = jax.lax.stop_gradient(x)
        k = pool(idx, inp, pos, idx.index_head_dim)
        q = idx.q_proj(jax.lax.stop_gradient(q_latent))
        q = q.reshape(batch, length, idx.index_n_heads, idx.index_head_dim).transpose(0, 2, 1, 3)
        q = idx.rotary_emb(q, pos, unsqueeze_dim=1)
        w = idx.weights_proj(inp).astype(jnp.float32) * idx.weights_scaling
        scores = idx._streamindex_scores_sharded(q, k, w)

        width = k.shape[1]
        rate = idx.compress_rate
        end = pos[:, :width*rate:rate] + rate
        valid = end[:, None, :] <= pos[:, :, None] + 1
        scores = jnp.where(valid, scores, -jnp.inf)
        if attention_mask is not None:
            seg = attention_mask[:, :, :width]
            scores += seg
            valid &= seg >= csa.DEFAULT_MASK_VALUE / 2

        if idx.index_topk < width:
            valid &= top(scores, idx.index_topk)
        valid = checkpoint_name(valid, "csa")
        mask = jnp.where(valid, 0.0, csa.DEFAULT_MASK_VALUE).astype(self.dtype)
        loss_scores = checkpoint_name(scores, "csa") if return_indexer_scores else None
        return pool(self, x, pos, self.head_dim)[:, :, None, :], mask[:, None, :, :], loss_scores

    cls.__call__ = compress
