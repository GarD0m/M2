"""Avoid gathering a dense validity mask at the selected CSA indices.

Training projections and pooling are adapted from MaxText 0.2.5,
copyright 2023–2026 Google LLC, Apache-2.0 (see LICENSES/MaxText-LICENSE.txt).
"""
from functools import wraps


def install_streamindex_mask(enabled=True):
    import jax
    import jax.numpy as jnp
    from maxtext.layers import attention_compressed as csa

    native = csa.DeepseekV4CSACompressor.__call__
    if getattr(native, "_llm5_dense_csa_mask", False):
        if not enabled:
            csa.DeepseekV4CSACompressor.__call__ = native.__wrapped__
        return
    if not enabled:
        return

    def pool(module, inputs, positions, width):
        compressed, _, _, _, _ = csa.compute_csa_prefill_chunk_pooling(
            kv=module.kv_proj(inputs),
            gate=module.gate_proj(inputs),
            seq_len=inputs.shape[1],
            batch_size=inputs.shape[0],
            compress_rate=module.compress_rate,
            position_ids=positions,
            position_bias=module.position_bias.value,
            kv_norm=module.kv_norm,
            rotary_emb=module.rotary_emb,
            head_dim=width,
            dtype=module.dtype,
        )
        return compressed

    @wraps(native)
    def compress(self, hidden_states, q_latent, position_ids, attention_mask=None,
                 model_mode=csa.MODEL_MODE_TRAIN, cache=None, indexer_cache=None,
                 use_indexer=True, return_indexer_scores=False):
        if (model_mode != csa.MODEL_MODE_TRAIN or cache is not None or indexer_cache is not None
                or not use_indexer or not self.indexer.config.use_csa_streamindex_kernel
                or hidden_states.shape[1] < self.compress_rate):
            return native(self, hidden_states, q_latent, position_ids, attention_mask, model_mode,
                          cache, indexer_cache, use_indexer, return_indexer_scores)

        indexer = self.indexer
        batch, length, _ = hidden_states.shape
        indexer_input = jax.lax.stop_gradient(hidden_states)
        keys = pool(indexer, indexer_input, position_ids, indexer.index_head_dim)
        query = indexer.q_proj(jax.lax.stop_gradient(q_latent))
        query = query.reshape(batch, length, indexer.index_n_heads, indexer.index_head_dim)
        query = indexer.rotary_emb(query.transpose(0, 2, 1, 3), position_ids, unsqueeze_dim=1)
        weights = indexer.weights_proj(indexer_input).astype(jnp.float32) * indexer.weights_scaling
        scores = indexer._streamindex_scores_sharded(query, keys, weights)

        width = keys.shape[1]
        rate = indexer.compress_rate
        block_positions = position_ids[:, :width * rate:rate]
        future = (block_positions[:, None, :] + rate) > (position_ids[:, :, None] + 1)
        scores = jnp.where(future, -jnp.inf, scores)
        invalid = future
        if attention_mask is not None:
            segment_mask = attention_mask[:, :, :width]
            scores = scores + segment_mask
            invalid = invalid | (segment_mask < csa.DEFAULT_MASK_VALUE / 2)

        # The consumer needs a dense mask. Intersect with validity in block order
        # after building membership, instead of gathering validity in top-k order.
        indices = jax.lax.top_k(scores, min(indexer.index_topk, width))[1]
        selected = jnp.any(indices[..., None] == jnp.arange(width), axis=-2)
        mask = jnp.where(selected & ~invalid, 0.0, csa.DEFAULT_MASK_VALUE).astype(self.dtype)
        compressed = pool(self, hidden_states, position_ids, self.head_dim)
        return compressed[:, :, None, :], mask[:, None, :, :], scores if return_indexer_scores else None

    compress._llm5_dense_csa_mask = True
    csa.DeepseekV4CSACompressor.__call__ = compress
