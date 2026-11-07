"""Reuse the vocabulary projection when differentiating each token tile.

Adapted from MaxText 0.2.5's vocabulary tiling, copyright 2025–2026 Google LLC,
Apache-2.0 (see LICENSES/MaxText-LICENSE.txt).
"""


def single_pass_tiled_loss(model, hidden_states, data, config):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from maxtext.utils import max_utils
    from maxtext.utils.vocabulary_tiling import _is_output_head_param_path

    graphdef, head, other, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)
    tiles = config.num_vocab_tiling

    def scan_loss(head, other, rest, hidden, labels, segmentation, differentiate):
        def project(params, x):
            local = nnx.merge(graphdef, params, other, rest, copy=True)
            return local.logits_from_hidden_states_for_vocab_tiling(x, True, "train")

        def body(carry, tile):
            losses, head_grad = carry
            x, targets, mask = tile
            targets = jax.nn.one_hot(targets, config.vocab_size)
            mask = mask != 0

            def terms(logits):
                loss, z_loss = max_utils.cross_entropy_with_logits(logits, targets, config.z_loss_multiplier)
                return jnp.sum(loss * mask), jnp.sum(z_loss * mask)

            if differentiate:
                logits, project_vjp = jax.vjp(project, head, x)
                _, loss_vjp = jax.vjp(terms, logits)
                (dlogits,) = loss_vjp((1.0, 0.0))
                head_update, dx = project_vjp(dlogits)
                head_grad = jax.tree.map(jnp.add, head_grad, head_update)
            else:
                logits = project(head, x)
                dx = None
            # Use the native primal loss for the metrics. It shares this tile's
            # projection and keeps the same reduction order as MaxText.
            loss, z_loss = terms(logits)
            return ((losses[0] + loss, losses[1] + z_loss), head_grad), dx

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
