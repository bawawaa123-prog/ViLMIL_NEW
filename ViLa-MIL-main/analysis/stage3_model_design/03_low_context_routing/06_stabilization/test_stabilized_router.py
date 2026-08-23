import unittest
import torch

from models.cross_scale_modules import HighRouter


def context(n=6, d=16):
    return {
        'high_parent_context': torch.randn(n, d),
        'high_parent_context_valid_mask': torch.tensor([True, True, False, True, False, True]),
        'high_valid_mask': torch.ones(n, dtype=torch.bool),
        'high_padding_ratio': torch.zeros(n),
    }


class StabilizedRouterTest(unittest.TestCase):
    def test_zero_init_baseline_and_unmapped(self):
        torch.manual_seed(1)
        high = torch.randn(6, 16)
        router = HighRouter(16, 8, stabilize=True, residual_ratio=0.10)
        out = router(high, context())
        torch.testing.assert_close(out, high, rtol=0, atol=0)
        torch.testing.assert_close(out[2], high[2], rtol=0, atol=0)

    def test_residual_cap_and_gradients(self):
        torch.manual_seed(2)
        high = torch.randn(6, 16, requires_grad=True)
        ctx = context()
        ctx['high_parent_context'].requires_grad_()
        router = HighRouter(16, 8, stabilize=True, residual_ratio=0.10)
        with torch.no_grad():
            router.context_projection.weight.normal_(0, 2)
        out = router(high, ctx)
        ratio = (out - high).norm(dim=1) / high.norm(dim=1).clamp_min(1e-8)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(bool((ratio[torch.tensor([0, 1, 3, 5])] <= 0.100001).all()))
        torch.testing.assert_close(out[2], high[2], rtol=0, atol=0)
        out.sum().backward()
        self.assertGreater(float(router.context_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(router.route_score[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(ctx['high_parent_context'].grad.abs().sum()), 0.0)

    def test_old_router_state_dict_compatibility(self):
        old = HighRouter(16, 8, stabilize=False)
        new = HighRouter(16, 8, stabilize=True, residual_ratio=0.10)
        result = new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


if __name__ == '__main__':
    unittest.main()
