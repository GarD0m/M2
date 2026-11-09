from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from llm5.cfg import short


def fast(shape, block_size):
    return tuple(shape) == (32, 1024, 4, 2048) and block_size == 128


def fused(config, permutations, saved, x, weights, cotangents):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from maxtext.kernels.mhc import common

    phi, h_pre = saved
    d_layer_input, (d_x_acc, d_h_post, d_residual) = cotangents
    batch, length, streams, width = x.shape
    tokens = batch * length
    block = min(config.bwd_block_size, 64)
    x_flat = x.reshape(tokens, streams, width)
    d_x_acc = d_x_acc.reshape(x_flat.shape)
    d_layer_input = d_layer_input.reshape(tokens, width)
    d_h_post = d_h_post.reshape(tokens, streams)
    d_residual = d_residual.reshape(tokens, streams, streams)

    coeffs = common.MhcCoeffParams(phi, weights.pre_scale, weights.pre_bias, weights.post_scale,
                                   weights.post_bias, weights.res_scale, weights.res_bias)

    def body(x_ref, h_pre_ref, d_layer_ref, d_acc_ref, d_post_ref, d_res_ref, perms_ref,
             coeff_refs, dx_ref, grad_refs):
        tile = x_ref[...]
        _, apply_vjp = jax.vjp(common.pre_apply, tile, h_pre_ref[...])
        dx_apply, d_h_pre = apply_vjp(d_layer_ref[...])

        def coeff_fn(stream_values, params):
            return common.mhc_coeffs(stream_values, params, perms_ref[...], rms_epsilon=config.rms_epsilon,
                                     pre_mapping_epsilon=config.pre_mapping_epsilon)

        params = jax.tree.map(lambda ref: ref[...], coeff_refs)
        _, coeff_vjp = jax.vjp(coeff_fn, tile, params)
        dx_coeff, d_coeffs = coeff_vjp(common.MhcCoeffOutputs(d_h_pre, d_post_ref[...], d_res_ref[...]))
        accumulated = (dx_apply.astype(jnp.float32) + d_acc_ref[...].astype(jnp.float32)).astype(dx_ref.dtype)
        dx_ref[...] = (dx_coeff.astype(jnp.float32) + accumulated.astype(jnp.float32)).astype(dx_ref.dtype)

        @pl.when(pl.program_id(0) == 0)
        def initialize():
            for ref in jax.tree.leaves(grad_refs):
                ref[...] = jnp.zeros_like(ref)

        for ref, grad in zip(jax.tree.leaves(grad_refs), jax.tree.leaves(d_coeffs)):
            ref[...] += grad.astype(jnp.float32)

    token_inputs = (x_flat, h_pre, d_layer_input, d_x_acc, d_h_post, d_residual)
    token_specs = tuple(common.token_block_spec(value.shape, block) for value in token_inputs)
    coeff_specs = jax.tree.map(lambda value: common.whole(value.shape), coeffs)
    grad_shapes = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, jnp.float32), coeffs)
    dims = common.MhcDims(tokens, streams, width, permutations.shape[0])
    apply_cost, coeff_cost = dims.pre_apply_bwd_cost(), dims.coeff_bwd_cost()
    stream_bytes = x.size * x.dtype.itemsize
    pipeline = pltpu.emit_pipeline(
        body,
        grid=(tokens // block,),
        in_specs=(*token_specs, common.whole(permutations.shape), coeff_specs),
        out_specs=(token_specs[0], coeff_specs),
        dimension_semantics=common.SEQUENTIAL_DIMENSION_SEMANTICS,
    )
    with common.tpu_mesh_context():
        dx, d_coeffs = pl.pallas_call(
            pipeline,
            out_shape=(jax.ShapeDtypeStruct(x_flat.shape, x.dtype), grad_shapes),
            in_specs=(*common.hbm_specs(len(token_inputs) + 1), common.hbm_specs(coeffs)),
            out_specs=(common.HBM_SPEC, common.hbm_specs(coeffs)),
            cost_estimate=pl.CostEstimate(
                flops=apply_cost.flops + coeff_cost.flops,
                transcendentals=coeff_cost.transcendentals,
                bytes_accessed=apply_cost.bytes_accessed + coeff_cost.bytes_accessed - 3 * stream_bytes,
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=config.vmem_limit_bytes),
            interpret=config.interpret,
            name="mhc_pre_bwd",
        )(*token_inputs, permutations, coeffs)

    _, phi_vjp = jax.vjp(common.fold_norm_scale, weights.norm_scale, weights.pre_alpha,
                         weights.post_alpha, weights.res_alpha)
    d_norm, d_pre_alpha, d_post_alpha, d_res_alpha = phi_vjp(d_coeffs.phi)
    return dx.reshape(x.shape), common.MhcWeights(
        norm_scale=d_norm,
        pre_alpha=d_pre_alpha,
        pre_bias=d_coeffs.pre_bias.astype(weights.pre_bias.dtype),
        pre_scale=d_coeffs.pre_scale.astype(weights.pre_scale.dtype),
        post_alpha=d_post_alpha,
        post_bias=d_coeffs.post_bias.astype(weights.post_bias.dtype),
        post_scale=d_coeffs.post_scale.astype(weights.post_scale.dtype),
        res_alpha=d_res_alpha,
        res_bias=d_coeffs.res_bias.astype(weights.res_bias.dtype),
        res_scale=d_coeffs.res_scale.astype(weights.res_scale.dtype),
    )


def row_dot(x, y):
    return jnp.einsum("bsd,bsd->bs", x, y, preferred_element_type=jnp.float32)[..., None]


def norm_fwd(x, scale, eps):
    inv = jax.lax.rsqrt(row_dot(x, x) / x.shape[-1] + eps)
    return x * inv.astype(x.dtype) * scale, (x, scale, inv)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def norm(x, scale, eps):
    return norm_fwd(x, scale, eps)[0]


def norm_bwd(eps, saved, dy):
    x, scale, inv = saved
    inv = inv.astype(x.dtype)
    y = x * inv
    dy_scale = dy * scale
    corr = (row_dot(y, dy_scale) / x.shape[-1]).astype(x.dtype)
    dx = (dy_scale - y * corr) * inv
    ds = jnp.einsum("bsd,bsd->d", dy, y, preferred_element_type=jnp.float32)
    return dx, ds.astype(scale.dtype)


norm.defvjp(norm_fwd, norm_bwd)


def head(self, x):
    precision = jax.lax.Precision(self.config.matmul_precision)

    def collapse(x):
        batch, length, streams, width = x.shape
        x = x.astype(jnp.bfloat16)
        flat = x.reshape(batch, length, streams * width)
        rms = row_dot(flat, flat) / flat.shape[-1]
        w = jnp.asarray(self.hc_fn[...], jnp.bfloat16)
        h = jnp.einsum("bsd,dk->bsk", flat, w, precision=precision, preferred_element_type=jnp.float32)
        h *= jax.lax.rsqrt(rms + self.eps)
        scale = jnp.asarray(self.hc_scale[...], jnp.float32)
        bias = jnp.asarray(self.hc_base[...], jnp.float32)
        gate = (jax.nn.sigmoid(h * scale + bias) + self.eps).astype(jnp.bfloat16)
        return jnp.einsum("bsk,bskd->bsd", gate, x, precision=precision,
                          preferred_element_type=jnp.float32).astype(self.dtype)

    return jax.checkpoint(collapse)(x)


def patch():
    from maxtext.kernels.mhc import mhc_kernels_bwd as bwd, mhc_kernels_fwd as fwd
    from maxtext.layers import mhc, nnx_decoders, normalizations

    class Norm(normalizations.RMSNorm):
        def __call__(self, inputs, out_sharding=None):
            if inputs.ndim != 3 or out_sharding is not None:
                return super().__call__(inputs, out_sharding)
            return norm(inputs, self.scale.get_value().astype(jnp.bfloat16), self.epsilon)

    native_perms = mhc.get_permutation_matrices

    def perms(k):
        with jax.ensure_compile_time_eval():
            return native_perms(k)

    native_policy = nnx_decoders.NNXDecoder.get_remat_policy
    save = jax.checkpoint_policies.save_only_these_names("moe_order", "csa", "mhc_pre")

    def policy(self):
        cfg = self.config
        base = native_policy(self)
        if (getattr(self, "remat_policy_override", None) is None
                and short(cfg.max_target_length, cfg.per_device_batch_size, self.mesh.size)):
            return lambda op, *args, **kw: save(op, *args, **kw) or base(op, *args, **kw)
        return base

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def pre(cfg, x, p, w):
        return fwd.pre_fwd(x, w, p, cfg)[0]

    def keep(value):
        return checkpoint_name(value, "mhc_pre")

    def pre_fwd(cfg, x, p, w):
        (branch, (streams, h, res)), saved = fwd.pre_fwd(x, w, p, cfg)
        return (keep(branch), (streams, keep(h), keep(res))), (jax.tree.map(keep, saved), x, p, w)

    def pre_bwd(cfg, state, dy):
        saved, x, p, w = state
        if fast(x.shape, cfg.bwd_block_size):
            dx, dw = fused(cfg, p, saved, x, w, dy)
        else:
            dx, dw = bwd.pre_op_bwd(cfg, p, (saved, (x, w)), dy)
        return dx, None, dw

    pre.defvjp(pre_fwd, pre_bwd)
    fwd._pre_op = pre
    mhc.get_permutation_matrices = perms
    mhc.DeepSeek4HyperHead.__call__ = head
    mhc.RMSNorm = Norm
    nnx_decoders.NNXDecoder.get_remat_policy = policy
