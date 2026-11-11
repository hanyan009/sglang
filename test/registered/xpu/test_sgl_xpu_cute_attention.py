"""sgl_xpu_cute diffusion attention backend: packed-sequence segmentation and selection.

``sgl_kernel.diffusion.sdp`` is replaced by torch SDPA, so no CUTE kernel is needed.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends import sgl_xpu_cute
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


def _sdp(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


class TestSglXpuCuteAttention(CustomTestCase):
    def _impl(self, **kw):
        args = dict(num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5)
        args.update(kw)
        return sgl_xpu_cute.SglXpuCuteImpl(**args)

    def test_enum_name(self):
        self.assertIs(
            AttentionBackendEnum["SGL_XPU_CUTE"],
            sgl_xpu_cute.SglXpuCuteBackend.get_enum(),
        )

    def test_rejects_unsupported_configs(self):
        for kw in (
            {"causal": True},
            {"head_size": 64},
            {"num_kv_heads": 1},
            {"softmax_scale": 0.5},
        ):
            with self.assertRaises(ValueError):
                self._impl(**kw)

    def test_varlen_segments_match_reference(self):
        impl = self._impl()
        q, k, v = (torch.randn(11, 2, 128) for _ in range(3))
        bounds = (0, 4, 4, 11)
        fake = mock.Mock(sdp=mock.Mock(side_effect=_sdp))
        with mock.patch.object(
            sgl_xpu_cute, "ensure_sgl_xpu_cute_available", return_value=fake
        ):
            out = impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=torch.tensor(bounds),
                max_seqlen=7,
                cu_seqlens_host=bounds,
            )
        self.assertEqual(fake.sdp.call_count, 2)  # empty segment skipped
        for a, b in ((0, 4), (4, 11)):
            torch.testing.assert_close(
                out[a:b], _sdp(q[None, a:b], k[None, a:b], v[None, a:b])[0]
            )

    def test_varlen_rejects_bad_bounds(self):
        impl = self._impl()
        x = torch.randn(5, 2, 128)
        with self.assertRaises(ValueError):
            impl.forward_varlen(
                x,
                x,
                x,
                cu_seqlens=torch.tensor([0, 4]),
                max_seqlen=4,
                cu_seqlens_host=(0, 4),
            )

    def test_unavailable_kernel_raises(self):
        with mock.patch.dict("sys.modules", {"sgl_kernel": None}):
            with self.assertRaises(RuntimeError):
                sgl_xpu_cute.ensure_sgl_xpu_cute_available()


if __name__ == "__main__":
    unittest.main()
