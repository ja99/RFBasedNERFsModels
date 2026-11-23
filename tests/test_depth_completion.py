import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from DepthCompletion import Completion, SupportCompletion, surface_features
from EchoKernel import _support
from Reconstruct import reconstruct


class DepthCompletionTests(unittest.TestCase):
    def test_blind_depth_only_completion_and_material_composition(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            acq = dict(grid_origin_cells=[0, 0, 0], grid_spacing_m=0.0045, sensor_layout="antenna,direction,timestep")
            settings = dict(
                acquisition=acq,
                grid_shape=[6, 6, 6],
                sensor_shape=[1, 1, 8],
                sensor_scale=1.0,
                sensor_offset=0.0,
            )
            camera = {"unit": "m"}
            z = np.ones((4, 4), dtype="f4") * 0.01
            free = np.zeros((6, 6, 6), bool)
            free[:2, 2:4, 2:4] = True
            surface = np.zeros_like(free)
            surface[2, 2:4, 2:4] = True
            expected = surface.copy()
            expected[3:5, 2:4, 2:4] = True
            optics = Mock()
            optics.observations.return_value = (free, surface)
            features, h, _, _ = surface_features(z, optics)
            network = SupportCompletion(len(features), 0)
            for p in network.parameters():
                p.data.zero_()
            checkpoint = root / "depth.pt"
            torch.save(
                dict(
                    version=1,
                    mode="depth",
                    settings=settings,
                    camera=camera,
                    pad=0.0,
                    d_norm=[torch.zeros(len(features)), torch.ones(len(features))],
                    r_norm=[torch.zeros(0), torch.ones(0)],
                    transforms=[],
                    state=[network.state_dict()],
                    y_scale=1.0,
                    y_mean=torch.from_numpy(_support(expected) - h),
                ),
                checkpoint,
            )
            scene = root / "measurements"
            scene.mkdir()
            (scene / "acquisition.json").write_text(json.dumps(acq))
            for filename in ("sensor_samples", "empty_sensor_samples"):
                np.save(scene / (filename + ".npy"), np.ones((1, 1, 8), dtype="f4"))
            camera_file = root / "camera.json"
            camera_file.write_text(json.dumps(camera))
            np.save(root / "depth.npy", z)
            options = dict(
                depth=root / "depth.npy",
                camera=camera_file,
                geometry_completion_model=checkpoint,
                depth_material_model=root / "material.npz",
            )
            material = Mock(mode="fusion", settings=settings, meta={"background": [1.0, 0.0]})
            material.predict.return_value = [4.0, 0.04]
            with (
                patch("DepthCompletion.registered_camera", return_value=optics),
                patch("DepthMaterial.DepthMaterial", return_value=material),
                patch("Reconstruct.release_asset", side_effect=AssertionError("unexpected weight fetch")),
            ):
                result = reconstruct(scene, root / "prediction.npz", depth_tolerance_mm=3.0, **options)
                optics.observations.assert_called_with(unittest.mock.ANY, tolerance_m=0.003)
                with np.load(result) as fields:
                    np.testing.assert_array_equal(fields["occupied"], expected)
                    np.testing.assert_array_equal(fields["sdf_cells"] < 0, expected)
                    self.assertEqual(str(fields["geometry_source"]), "depth_completion")
                    self.assertEqual(fields["geometry_beam_indices"].size, 0)
                    np.testing.assert_array_equal(fields["material_beam_indices"], [0])
                    np.testing.assert_allclose(fields["permittivity"][expected], 4.0)
                    np.testing.assert_allclose(fields["permittivity"][~expected], 1.0)
                with self.assertRaisesRegex(ValueError, "separate"):
                    reconstruct(scene, checkpoint, **options)
                with self.assertRaisesRegex(ValueError, "calibration differs"):
                    Completion(checkpoint).predict(z, {"unit": "mm"})
                optics.observations.return_value = (free, np.zeros_like(free))
                with self.assertRaisesRegex(ValueError, "positive depth return"):
                    Completion(checkpoint).predict(z, camera)
