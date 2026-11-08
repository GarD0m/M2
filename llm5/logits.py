"""Reuse the vocabulary projection when differentiating each token tile.

Adapted from MaxText 0.2.5's vocabulary tiling, copyright 2025–2026 Google LLC,
Apache-2.0 (see LICENSES/MaxText-LICENSE.txt).
"""

from llm5.plan import GROUPED_LOGITS_TILES


def single_pass_tiled_loss(model, hidden_states, data, config):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils.vocabulary_tiling import _is_output_head_param_path

    graphdef, head, other, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)
    tiles = config.num_vocab_tiling
    z_multiplier = config.z_loss_multiplier
    keep_bf16_logits = (
        config.dtype == jnp.bfloat16 and not config.logits_dot_in_fp32
        and not config.normalize_embedding_logits and not config.final_logits_soft_cap
    )
    device_kind = model.mesh.devices.flat[0].device_kind.casefold()
    tokens = hidden_states.shape[0] * hidden_states.shape[1]
    group_gradients = (
        keep_bf16_logits and tiles == GROUPED_LOGITS_TILES.get(device_kind)
        and hidden_states.shape[1] <= 1024 and tokens <= 32768
        and tokens * config.vocab_size <= 2 * 1024**3
        and hasattr(getattr(model, "decoder", None), "decoder_norm")
        and hasattr(model.token_embedder, "embedding")
        and isinstance(model.token_embedder.embedding[...], jax.Array)
    )

    def tile_loss(logits, targets, mask, differentiate):
        mask = mask != 0
        row_max = logits.max(axis=-1).astype(jnp.float32)
        exp_shifted = jnp.exp(logits.astype(jnp.float32) - row_max[:, None])
        sum_exp = exp_shifted.sum(axis=-1)
        log_sum = jnp.log(sum_exp)
        log_z = row_max + log_sum
        vocab = logits.shape[-1]
        valid_targets = (targets >= 0) & (targets < vocab)
        target_logits = jnp.take_along_axis(logits, jnp.clip(targets, 0, vocab - 1)[:, None], axis=-1)[:, 0]
        z_loss = z_multiplier * jax.lax.square(log_z)
        xent = jnp.where(valid_targets, log_sum - (target_logits.astype(jnp.float32) - row_max), 0.0) + z_loss
        dlogits = None
        if differentiate:
            scale = (1 + 2 * z_multiplier * log_z)[:, None]
            target_matches = targets[:, None] == jnp.arange(vocab)
            dlogits = ((scale * exp_shifted / sum_exp[:, None] - target_matches) * mask[:, None]).astype(logits.dtype)
        return (jnp.sum(xent * mask), jnp.sum(z_loss * mask)), dlogits

    def scan_loss(head, other, rest, hidden, labels, segmentation, differentiate):
        def project(params, x):
            local = nnx.merge(graphdef, params, other, rest, copy=True)
            logits = local.logits_from_hidden_states_for_vocab_tiling(x, True, "train")
            # The tied projection already rounded to BF16 before MaxText's
            # FP32 cast. Keep the label gather on the smaller representation.
            return logits.astype(jnp.bfloat16) if keep_bf16_logits else logits

        def body(carry, tile):
            losses, head_grad = carry
            x, targets, mask = tile
            if differentiate:
                logits, project_vjp = jax.vjp(project, head, x)
            else:
                logits = project(head, x)
            (loss, z_loss), dlogits = tile_loss(logits, targets, mask, differentiate)
            if differentiate:
                head_update, dx = project_vjp(dlogits)
                head_grad = jax.tree.map(jnp.add, head_grad, head_update)
            else:
                dx = None
            return ((losses[0] + loss, losses[1] + z_loss), head_grad), dx

        def grouped_body(carry, group):
            losses, head_grad = carry
            x, targets, mask = group

            def prepare(params, inputs):
                local = nnx.merge(graphdef, params, other, rest, copy=True)
                return local.decoder.decoder_norm(inputs), local.token_embedder.embedding[...].astype(jnp.bfloat16)

            flat_x = x.reshape(-1, x.shape[-1])
            (normalized, embedding), prepare_vjp = jax.vjp(prepare, head, flat_x)

            def one_tile(tile):
                y, ids, valid = tile
                logits = jnp.dot(y, embedding.T, preferred_element_type=jnp.bfloat16)
                return tile_loss(logits, ids, valid, True)

            # Keep the FP32 softmax workspace at one tile. Combine four BF16
            # cotangent tiles for one input GEMM and one embedding update.
            (loss, z_loss), dlogits = jax.lax.map(one_tile, (normalized.reshape(x.shape), targets, mask))
            dlogits = dlogits.reshape(flat_x.shape[0], -1)
            dy = jnp.dot(dlogits, embedding, preferred_element_type=normalized.dtype)
            dembed = jnp.dot(dlogits.T, normalized, preferred_element_type=embedding.dtype)
            head_update, dx = prepare_vjp((dy, dembed))
            head_grad = jax.tree.map(jnp.add, head_grad, head_update)
            return ((losses[0] + loss.sum(), losses[1] + z_loss.sum()), head_grad), dx

        zero = jnp.zeros((), dtype=jnp.float32)
        head_grad = jax.tree.map(jnp.zeros_like, head) if differentiate else None
        inputs = (hidden.reshape(tiles, -1, hidden.shape[-1]),
                  labels.reshape(tiles, -1), segmentation.reshape(tiles, -1))
        if differentiate and group_gradients:
            grouped = jax.tree.map(lambda value: value.reshape(tiles // 4, 4, *value.shape[1:]), inputs)
            return jax.lax.scan(grouped_body, ((zero, zero), head_grad), grouped)
        return jax.lax.scan(body, ((zero, zero), head_grad), inputs)

    @jax.custom_vjp
    def tiled_loss(head, other, rest, hidden, labels, segmentation):
        (losses, _), _ = scan_loss(head, other, rest, hidden, labels, segmentation, False)
        return losses

    def forward(head, other, rest, hidden, labels, segmentation):
        (losses, dhead), dx = scan_loss(head, other, rest, hidden, labels, segmentation, True)
        return losses, (dhead, dx.reshape(hidden.shape), other, rest)

    def backward(residuals, cotangents):
        dhead, dx, other, rest = residuals
        scale, _ = cotangents  # z-loss is already included in the first output.
        dhead = jax.tree.map(lambda g: (g * scale).astype(g.dtype), dhead)
        dx = (dx * scale).astype(dx.dtype)
        # Explicit zero pytrees preserve the cotangent layout of scanned parameters.
        return dhead, jax.tree.map(jnp.zeros_like, other), jax.tree.map(jnp.zeros_like, rest), dx, None, None

    tiled_loss.defvjp(forward, backward)
    return tiled_loss(head, other, rest, hidden_states, data["targets"], data["targets_segmentation"])
