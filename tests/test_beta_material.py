import unittest

import numpy as np
import torch

from BetaGaussian import beta_gaussian_loss
from BetaMaterial import calibrate_variance, physical_moments


class BetaMaterialTests(unittest.TestCase):
    def test_variance_weight_has_no_gradient(self):
        for beta in (0.0, 0.5, 1.0):
            mu = torch.tensor([1.0], requires_grad=True)
            variance = torch.tensor([4.0], requires_grad=True)
            y = torch.tensor([3.0])
            loss = beta_gaussian_loss(mu, variance, y, beta)
            loss.backward()
            self.assertAlmostEqual(mu.grad.item(), -2 * 4 ** (beta - 1))
            self.assertEqual(variance.grad.item(), 0.0)

    def test_physical_ensemble_variance(self):
        class Example:
            def __init__(self, mean):
                self.mean = mean
                self.target_mean = torch.tensor([3.0, 7.0])
                self.target_scale = torch.tensor([2.0, 3.0])

            def __call__(self, _unused):
                return torch.tensor([[self.mean, self.mean]]), torch.tensor(
                    [[2.0, 2.0]]
                )

        mu, variance = physical_moments([Example(-1.0), Example(1.0)], [None])
        np.testing.assert_array_equal(mu, [[3.0, 7.0]])
        np.testing.assert_array_equal(variance, [[12.0, 27.0]])

    def test_calibration_estimates_residual_scale(self):
        mean, y = np.zeros((2, 2)), np.array([[2.0, 3.0], [4.0, 1.0]])
        var = np.array([[1.0, 1.0], [4.0, 1.0]])
        constant, scale = calibrate_variance(mean, var, y)
        np.testing.assert_allclose(constant, [10.0, 5.0])
        np.testing.assert_allclose(scale, [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
