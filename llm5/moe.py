from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

def gather(x, ids):
    return x.at[ids].get(mode="promise_in_bounds", wrap_negative_indices=False)


def route_fwd(x, ids):
    k = ids.size // x.shape[0]
    perm = checkpoint_name(jnp.argsort(ids.reshape(-1)), "moe_routing_maps")
    return gather(x, perm // k), perm.reshape(-1, k)


@jax.custom_vjp
def route(x, ids):
    return route_fwd(x, ids)[0]


def route_bwd(perm, dy):
    y = gather(dy, jnp.argsort(perm.reshape(-1)))
    return y.reshape(*perm.shape, *dy.shape[1:]).sum(axis=1), None


route.defvjp(route_fwd, route_bwd)


def combine_fwd(x, perm, w, precision):
    ordered = gather(x, jnp.argsort(perm)).reshape(*w.shape, x.shape[-1])
    ordered = checkpoint_name(ordered, "moe_order")
    y = jnp.einsum("tke,tk->te", ordered, w, precision=precision)
    return y, (ordered, perm, w)


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def combine(x, perm, w, precision):
    return combine_fwd(x, perm, w, precision)[0]


def combine_bwd(precision, saved, dy):
    ordered, perm, w = saved
    dx = gather(dy, perm // w.shape[1]) * gather(w.reshape(-1), perm)[:, None]
    dw = jnp.einsum("te,tke->tk", dy, ordered, precision=precision)
    return dx, None, dw


combine.defvjp(combine_fwd, combine_bwd)


def patch():
    from maxtext.layers import moe
    from maxtext.models import deepseek4

    def unpermute(self, x, perm, w, batch_size, sequence_length, use_custom_sort_vjp=True,
                  group_sizes=None, topk_argsort_indices=None):
        w = w.reshape(-1, self.num_experts_per_tok)
        with jax.named_scope("weight_sum"):
            y = combine(x, perm, w, jax.lax.Precision(self.config.matmul_precision))
        return y.reshape(batch_size, sequence_length, -1).astype(self.dtype)

    class Experts(nnx.Module):
        def __init__(self, config, **kwargs):
            self.MoeBlock_0 = moe.RoutedMoE(config=config, num_experts=config.num_experts,
                                           num_experts_per_tok=config.num_experts_per_tok,
                                           intermediate_dim=config.moe_mlp_dim, **kwargs)

        def __call__(self, inputs, input_ids=None):
            return self.MoeBlock_0(inputs=inputs, input_ids=input_ids)

    moe._route_activations = route
    moe.RoutedMoE.unpermute = unpermute
    deepseek4.moe = SimpleNamespace(RoutedAndSharedMoE=Experts)
