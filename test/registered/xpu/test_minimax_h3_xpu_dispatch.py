"""MiniMax-H3 XPU fusions: dispatch to sgl_kernel.h3_fused and fall back otherwise.

The kernel entry points are mocked, so these tests do not need a kernel build that
ships ``sgl_kernel.h3_fused``.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

_MOD = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"


@unittest.skipUnless(torch.xpu.is_available(), "XPU not available")
class TestMiniMaxH3XPUDispatch(CustomTestCase):
    def setUp(self):
        from sglang.multimodal_gen.runtime.models.dits import minimax_h3

        self.m = minimax_h3
        self.x = torch.randn(8, 64, dtype=torch.bfloat16, device="xpu")

    def test_silu_mul_uses_kernel_when_enabled(self):
        sentinel = torch.empty(8, 32, dtype=torch.bfloat16, device="xpu")
        with (
            mock.patch(f"{_MOD}.can_use_h3_silu_mul", return_value=True),
            mock.patch(f"{_MOD}.h3_silu_mul", return_value=sentinel) as kernel,
        ):
            out = self.m._silu_mul(self.x, reuse_input=False, fused_xpu=True)
        kernel.assert_called_once()
        self.assertIs(out, sentinel)

    def test_silu_mul_falls_back(self):
        expected = torch.nn.functional.silu(self.x[:, :32]) * self.x[:, 32:]
        with (
            mock.patch(f"{_MOD}.can_use_h3_silu_mul", return_value=True),
            mock.patch(f"{_MOD}.h3_silu_mul") as kernel,
        ):
            out = self.m._silu_mul(self.x, reuse_input=False, fused_xpu=False)
        kernel.assert_not_called()
        torch.testing.assert_close(out, expected, atol=0.05, rtol=0.05)

    def test_silu_mul_falls_back_when_kernel_missing(self):
        with (
            mock.patch(f"{_MOD}.can_use_h3_silu_mul", None),
            mock.patch(f"{_MOD}.h3_silu_mul", None),
        ):
            out = self.m._silu_mul(self.x, reuse_input=False, fused_xpu=True)
        self.assertEqual(out.shape, (8, 32))

    def test_indexed_gate_dispatch(self):
        gate = torch.randn(3, 64, dtype=torch.bfloat16, device="xpu")
        idx = torch.zeros(8, dtype=torch.int64, device="xpu")
        sentinel = torch.empty_like(self.x)
        with (
            mock.patch(f"{_MOD}.can_use_h3_indexed_gate", return_value=True),
            mock.patch(f"{_MOD}.h3_indexed_gate", return_value=sentinel) as kernel,
        ):
            out = self.m._modulate_gate(
                self.x, gate, self.x, idx, dtype=torch.bfloat16, fused=True
            )
        kernel.assert_called_once()
        self.assertIs(out, sentinel)
        with (
            mock.patch(f"{_MOD}.can_use_h3_indexed_gate", return_value=True),
            mock.patch(f"{_MOD}.h3_indexed_gate") as kernel,
        ):
            self.m._modulate_gate(
                self.x.clone(), gate, self.x, idx, dtype=torch.bfloat16, fused=False
            )
        kernel.assert_not_called()

    def test_rmsnorm_adaln_dispatch(self):
        norm = torch.nn.RMSNorm(64, eps=1e-5, device="xpu", dtype=torch.bfloat16)
        shift = torch.randn(3, 64, dtype=torch.bfloat16, device="xpu")
        idx = torch.zeros(8, dtype=torch.int64, device="xpu")
        sentinel = torch.empty_like(self.x)
        with (
            mock.patch(f"{_MOD}.can_use_h3_rmsnorm_indexed_adaln", return_value=True),
            mock.patch(
                f"{_MOD}.h3_rmsnorm_indexed_adaln", return_value=sentinel
            ) as kernel,
        ):
            out = self.m._modulate_rmsnorm_scale_shift(
                self.x, norm, shift, shift, idx, dtype=torch.bfloat16, fused=True
            )
        kernel.assert_called_once()
        self.assertIs(out, sentinel)
        with (
            mock.patch(f"{_MOD}.can_use_h3_rmsnorm_indexed_adaln", return_value=False),
            mock.patch(f"{_MOD}.h3_rmsnorm_indexed_adaln") as kernel,
        ):
            self.m._modulate_rmsnorm_scale_shift(
                self.x, norm, shift, shift, idx, dtype=torch.bfloat16, fused=True
            )
        kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
