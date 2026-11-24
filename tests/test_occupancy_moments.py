import unittest
from types import SimpleNamespace

import numpy as np
from EchoKernel import DIRECTIONS
from OccupancyMoments import OccupancyMoments


class TestOccupancyMoments(unittest.TestCase):
    def test_samples_obey_geometry_and_carry_joint_surface_error(self):
        free = np.zeros((11, 11, 11), dtype=bool)
        free[5, 5, 7] = True
        hit = np.zeros_like(free)
        hit[5, 5, 3] = True
        model = object.__new__(OccupancyMoments)
        model.model = SimpleNamespace(optics=SimpleNamespace(observations=lambda depth: (free, hit)),
                                      meta={"pad": 0.})
        visible = DIRECTIONS @ np.array([5, 5, 5])
        obs = dict(visible_support_cells=visible, expected_support_cells=visible+2,
                   variance_support_cells2=np.ones(len(DIRECTIONS)))
        model.standardized = np.stack((-np.ones(len(DIRECTIONS)), np.ones(len(DIRECTIONS))))
        model.residuals = np.zeros_like(model.standardized)
        p = model.probability(obs, None)
        self.assertEqual(p[5, 5, 7], 0)
        self.assertEqual(p[5, 5, 3], 1)
        self.assertEqual(p[5, 5, 5], 1)
        self.assertEqual(p[5, 7, 5], .5)
        self.assertEqual(p[5, 3, 5], .5)
        self.assertFalse((model.probability(obs, None, constant=True) == .5).any())
