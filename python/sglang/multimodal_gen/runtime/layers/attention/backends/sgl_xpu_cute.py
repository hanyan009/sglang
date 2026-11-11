# SPDX-License-Identifier: Apache-2.0
"""sgl-kernel-xpu's single-dispatch CUTE attention for packed H3 sequences."""

import math

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def ensure_sgl_xpu_cute_available():
    try:
        from sgl_kernel import diffusion
    except ImportError as exc:
        raise RuntimeError(
            "sgl-kernel-xpu CUTE attention requires sgl_kernel.diffusion"
        ) from exc
    if not diffusion.is_available():
        raise RuntimeError(
            "sgl-kernel-xpu CUTE attention native extension is unavailable"
        )
    return diffusion


class SglXpuCuteBackend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes():
        return [128]

    @staticmethod
    def get_enum():
        return AttentionBackendEnum.SGL_XPU_CUTE

    @staticmethod
    def get_impl_cls():
        return SglXpuCuteImpl

    @staticmethod
    def get_metadata_cls():
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return FlashAttentionMetadataBuilder


class SglXpuCuteImpl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        causal,
        softmax_scale,
        num_kv_heads=None,
        prefix="",
        **extra_impl_args,
    ):
        if causal or head_size != 128 or num_kv_heads not in (None, num_heads):
            raise ValueError("sgl XPU CUTE supports noncausal D128 self-attention only")
        if not math.isclose(softmax_scale, 128**-0.5, rel_tol=0, abs_tol=1e-12):
            raise ValueError("sgl XPU CUTE requires 1/sqrt(128) softmax scale")

    def _sdp(self, query, key, value):
        return ensure_sgl_xpu_cute_available().sdp(query, key, value)

    def forward(self, query, key, value, attn_metadata=None):
        return self._sdp(query, key, value)

    def forward_varlen(
        self,
        query,
        key,
        value,
        *,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_host=None,
    ):
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(v) for v in cu_seqlens.tolist())
        )
        if (
            not bounds
            or bounds[0] != 0
            or bounds[-1] != query.shape[0]
            or any(stop < start for start, stop in zip(bounds[:-1], bounds[1:]))
        ):
            raise ValueError(f"Invalid packed attention bounds: {bounds}")
        out = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if stop > start:
                out[start:stop].copy_(
                    self._sdp(
                        query[start:stop].unsqueeze(0),
                        key[start:stop].unsqueeze(0),
                        value[start:stop].unsqueeze(0),
                    ).squeeze(0)
                )
        return out
