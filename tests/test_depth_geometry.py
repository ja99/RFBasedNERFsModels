import unittest

import numpy as np

from DepthGeometry import DEFAULT_SIMULATOR, constrain, registered_camera
from EchoKernel import occupancy_sdf


class DepthGeometryTests(unittest.TestCase):
    def test_reconstructed_voxel_sdf(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        sdf = occupancy_sdf(mask)
        self.assertAlmostEqual(sdf[2, 2, 2], -1.5)
        self.assertAlmostEqual(sdf[1, 1, 1], -.5)
        self.assertAlmostEqual(sdf[0, 0, 0], np.sqrt(.75))
        self.assertTrue(np.isposinf(occupancy_sdf(mask & False)).all())

    @unittest.skipUnless((DEFAULT_SIMULATOR / 'DepthCamera.py').is_file(), 'optional camera checkout required')
    def test_geometric_projection_preserves_occluded_prior_and_checks_registration(self):
        shape = (10, 10, 10)
        acquisition = dict(grid_origin_cells=[1, 3, 4], grid_spacing_m=.01, antenna_positions_cells=[[1, 7, 8]])
        spec = dict(depth_camera_version=1, intrinsics=dict(width=3, height=3, hfov_deg=90., fx=1.5, fy=1.5,
                      cx=1., cy=1.), position_m=[.01, .07, .08], camera_to_world_rotation=[[0,0,1],[-1,0,0],[0,-1,0]],
                    grid_origin_cells=[1,3,4], grid_spacing_m=.01, crop_shape_cells=list(shape),
                    depth_quantity="optical_axis_distance_m", no_return_value=0.,
                    scene_model="opaque_unit_voxels_centered_on_integer_world_grid_coordinates")
        # Compute fx by the exact calibration formula used by the physical camera.
        spec["intrinsics"]["fx"] = spec["intrinsics"]["fy"] = 3 / (2 * np.tan(np.pi / 4))
        camera = registered_camera(acquisition, shape, spec)
        front = np.zeros(shape, dtype=bool)
        front[4] = True
        depth = camera.render(front)
        prior = front.copy()
        prior[7, 4:6, 4:6] = True
        self.assertTrue(np.array_equal(camera.render(prior), depth))
        constraints = camera.observations(depth)
        np.testing.assert_array_equal(constrain(prior, *constraints)[5:], prior[5:])
        np.testing.assert_array_equal(constrain(front, *constraints)[5:], front[5:])
        with self.assertRaisesRegex(ValueError, "calibration"):
            registered_camera(acquisition, shape, {**spec, "position_m": [0., 0., 0.]})
        with self.assertRaisesRegex(ValueError, "calibration"):
            registered_camera(acquisition, (9,10,10), spec)
