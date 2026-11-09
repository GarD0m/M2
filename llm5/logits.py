import math

import jax
import jax.numpy as jnp
from flax import nnx

from llm5.cfg import short, tiles


def loss(model, hidden, data, cfg, count):
    from maxtext.utils.vocabulary_tiling import _is_output_head_param_path

    graph, head, other, rest = nnx.split(model, _is_output_head_param_path, nnx.Param, ...)
    z = cfg.z_loss_multiplier
    grouped = count == 16 and short(hidden.shape[1], hidden.shape[0], model.mesh.size)

    def softmax(logits, ids, mask, grad):
        mask = mask != 0
        mx = logits.max(axis=-1).astype(jnp.float32)
        exp = jnp.exp(logits.astype(jnp.float32) - mx[:, None])
        total = exp.sum(axis=-1)
        log_total = jnp.log(total)
        log_z = mx + log_total
        vocab = logits.shape[-1]
        valid = (ids >= 0) & (ids < vocab)
        target = jnp.take_along_axis(logits, jnp.clip(ids, 0, vocab-1)[:, None], axis=-1)[:, 0]
        zl = z * jax.lax.square(log_z)
        ce = jnp.where(valid, log_total - (target.astype(jnp.float32) - mx), 0)
        dy = None
        if grad:
            scale = jnp.where(mask, (1 + 2*z*log_z) / total, 0)
            labels = jnp.where(mask, ids, -1)[:, None] == jnp.arange(vocab)
            dy = (exp * scale[:, None] - labels).astype(logits.dtype)
        return (jnp.sum((ce + zl) * mask), jnp.sum(zl * mask)), dy

    def run(head, other, rest, x, ids, mask, grad):
        def project(params, x):
            net = nnx.merge(graph, params, other, rest, copy=True)
            return net.logits_from_hidden_states_for_vocab_tiling(x, True, "train").astype(jnp.bfloat16)

        def prepare(params, x):
            net = nnx.merge(graph, params, other, rest, copy=True)
            return net.decoder.decoder_norm(x), net.token_embedder.embedding[...].astype(jnp.bfloat16)

        def step(state, tile):
            ce, zl, dh = state
            x, ids, mask = tile
            if grad:
                logits, vjp = jax.vjp(project, head, x)
            else:
                logits = project(head, x)
            (a, b), dy = softmax(logits, ids, mask, grad)
            dx = None
            if grad:
                update, dx = vjp(dy)
                dh = jax.tree.map(jnp.add, dh, update)
            return (ce+a, zl+b, dh), dx

        def group(state, group):
            ce, zl, dh = state
            x, ids, mask = group
            flat = x.reshape(-1, x.shape[-1])
            (y, w), vjp = jax.vjp(prepare, head, flat)

            def tile(args):
                y, ids, mask = args
                logits = jnp.dot(y, w.T, preferred_element_type=jnp.bfloat16)
                return softmax(logits, ids, mask, True)

            (a, b), dy = jax.lax.map(tile, (y.reshape(x.shape), ids, mask))
            dy = dy.reshape(flat.shape[0], -1)
            dx = jnp.dot(dy, w, preferred_element_type=y.dtype)
            dw = jnp.dot(dy.T, y, preferred_element_type=w.dtype)
            update, dx = vjp((dx, dw))
            return (ce+a.sum(), zl+b.sum(), jax.tree.map(jnp.add, dh, update)), dx

        x = x.reshape(count, -1, x.shape[-1])
        inputs = (x, ids.reshape(x.shape[:2]), mask.reshape(x.shape[:2]))
        zero = jnp.zeros((), jnp.float32)
        state = (zero, zero, jax.tree.map(jnp.zeros_like, head) if grad else None)
        if grad and grouped:
            inputs = jax.tree.map(lambda a: a.reshape(count // 4, 4, *a.shape[1:]), inputs)
            return jax.lax.scan(group, state, inputs)
        return jax.lax.scan(step, state, inputs)

    @jax.custom_vjp
    def calc(head, other, rest, x, ids, mask):
        (ce, zl, _), _ = run(head, other, rest, x, ids, mask, False)
        return ce, zl

    def fwd(head, other, rest, x, ids, mask):
        (ce, zl, dh), dx = run(head, other, rest, x, ids, mask, True)
        return (ce, zl), (dh, dx.reshape(x.shape), other, rest)

    def bwd(saved, dy):
        dh, dx, other, rest = saved
        scale = dy[0]
        return (jax.tree.map(lambda g: (g * scale).astype(g.dtype), dh),
                jax.tree.map(jnp.zeros_like, other), jax.tree.map(jnp.zeros_like, rest),
                (dx * scale).astype(dx.dtype), None, None)

    calc.defvjp(fwd, bwd)
    return calc(head, other, rest, hidden, data["targets"], data["targets_segmentation"])


def patch(train):
    from maxtext.configs.pyconfig import HyperParameters

    native = train.vocab_tiling_nnx_loss

    def tiled(model, hidden, data, config, is_train):
        devices = model.mesh.size
        tokens = math.prod(hidden.shape[:-1]) // devices
        limit = config.num_vocab_tiling
        if not is_train:
            limit = max(1, math.ceil(limit * min(1, config.eval_per_device_batch_size / config.per_device_batch_size)))
        count = tiles(tokens, limit)
        if is_train and devices == 1:
            return loss(model, hidden, data, config, count)
        if count != config.num_vocab_tiling:
            config = HyperParameters(config._pydantic_config.model_copy(update={"num_vocab_tiling": count}))
        return native(model, hidden, data, config, is_train)

    train.vocab_tiling_nnx_loss = tiled
