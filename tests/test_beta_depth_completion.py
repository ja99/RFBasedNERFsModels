"""Regression tests for directional uncertainty semantics."""

import unittest

import numpy as np
import torch
from torch import nn

from BetaDepthCompletion import DistributionNet, beta_loss, gaussian_nll, moments


class ConstantHead(nn.Module):
    def __init__(self, mean, variance):
        super().__init__()
        self.mean = mean
        self.variance = variance

    def forward(self, depth, rf):
        return torch.full_like(depth, self.mean), torch.full_like(depth, self.variance)


class SupportDistributionTests(unittest.TestCase):
    def test_total_variance_and_units(self):
        x = (torch.zeros(4, 2), torch.zeros(4, 1))
        mu, var = moments(
            [ConstantHead(-1.0, 0.25), ConstantHead(1.0, 0.75)],
            x,
            {"y_scale": 3.0, "y_mean": torch.tensor([2.0, 8.0])},
        )
        np.testing.assert_array_equal(mu, np.tile([2.0, 8.0], (4, 1)))
        np.testing.assert_allclose(var, (1 + 0.5) * 9.0)

    def test_beta_one_preserves_squared_error_mean_gradient(self):
        y = torch.tensor([2.0])
        m = torch.tensor([-0.5], requires_grad=True)
        v = torch.tensor([0.25], requires_grad=True)
        beta_loss(m, v, y, 1.0).backward()
        self.assertAlmostEqual(m.grad.item(), -2.5)
        self.assertAlmostEqual(v.grad.item(), 0.5 * (1 - 2.5**2 / 0.25))

    def test_selection_uses_unweighted_density(self):
        self.assertAlmostEqual(float(gaussian_nll(0.0, 0.0, 1.0)), 0.5 * np.log(2 * np.pi))
        self.assertGreater(gaussian_nll(3.0, 0.0, 0.01), gaussian_nll(3.0, 0.0, 1.0))

    def test_model_heads_and_positive_variance(self):
        model = DistributionNet(4, 2)
        m, v = model(torch.zeros(3, 4), torch.ones(3, 2))
        self.assertEqual(m.shape, (3, 98))
        self.assertEqual(v.shape, m.shape)
        self.assertTrue(torch.all(v > 0))


if __name__ == "__main__":
    unittest.main()
