"""Reuse the vocabulary projection when differentiating each token tile.

Adapted from MaxText 0.2.5's vocabulary tiling, copyright 2025–2026 Google LLC,
Apache-2.0 (see LICENSES/MaxText-LICENSE.txt).
"""


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
            mask = mask != 0
            if differentiate:
                logits, project_vjp = jax.vjp(project, head, x)
            else:
                logits = project(head, x)

            row_max = logits.max(axis=-1).astype(jnp.float32)
            exp_shifted = jnp.exp(logits.astype(jnp.float32) - row_max[:, None])
            sum_exp = exp_shifted.sum(axis=-1)
            log_sum = jnp.log(sum_exp)
            log_z = row_max + log_sum
            vocab = logits.shape[-1]
            valid_targets = (targets >= 0) & (targets < vocab)
            # Only one logit per row contributes to the label term. Gather
            # before centering to avoid reducing a dense one-hot target.
            target_logits = jnp.take_along_axis(logits, jnp.clip(targets, 0, vocab - 1)[:, None], axis=-1)[:, 0]
            target_logits = target_logits.astype(jnp.float32)
            z_loss = z_multiplier * jax.lax.square(log_z)
            xent = jnp.where(valid_targets, log_sum - (target_logits - row_max), 0.0) + z_loss

            if differentiate:
                scale = (1 + 2 * z_multiplier * log_z)[:, None]
                target_matches = targets[:, None] == jnp.arange(vocab)
                dlogits = (scale * exp_shifted / sum_exp[:, None] - target_matches) * mask[:, None]
                head_update, dx = project_vjp(dlogits.astype(logits.dtype))
                head_grad = jax.tree.map(jnp.add, head_grad, head_update)
            else:
                dx = None
            return ((losses[0] + jnp.sum(xent * mask), losses[1] + jnp.sum(z_loss * mask)), head_grad), dx

        zero = jnp.zeros((), dtype=jnp.float32)
        head_grad = jax.tree.map(jnp.zeros_like, head) if differentiate else None
        inputs = (hidden.reshape(tiles, -1, hidden.shape[-1]),
                  labels.reshape(tiles, -1), segmentation.reshape(tiles, -1))
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
