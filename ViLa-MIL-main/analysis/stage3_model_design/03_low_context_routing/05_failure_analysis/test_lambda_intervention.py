import unittest
import torch


class LambdaInterventionTest(unittest.TestCase):
    def test_scale_formula_and_zero_identity(self):
        high = torch.randn(5, 8)
        residual = torch.randn(5, 8)
        for lam in (0.0, 0.1, 0.25, 0.5, 1.0):
            routed = high + lam * residual
            if lam == 0:
                torch.testing.assert_close(routed, high, rtol=0, atol=0)
            else:
                torch.testing.assert_close(routed - high, lam * residual)


if __name__ == '__main__':
    unittest.main()
