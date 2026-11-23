import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from NeuralDepthMaterial import MaterialNetwork, NeuralDepthMaterial, sensor_features


class NeuralMaterialTest(unittest.TestCase):
    def test_checkpoint_and_measured_pair_inference(self):
        rng = np.random.default_rng(31)
        arrays = rng.normal(0, .1, (16, 1, 128)).astype("f4")
        camera = {"intrinsics": dict(width=16, height=16, fx=8., fy=8., cx=7.5, cy=7.5)}
        depth = np.zeros((16, 16), dtype="f4")
        depth[5:9, 6:11] = .03
        acquisition = dict(antenna_positions_cells=[[0, i, j] for i in range(4) for j in range(4)],
                           frequency_hz=2.4e9, time_step_s=8.6e-12, sample_stride=2,
                           sensor_layout="antenna,direction,timestep")
        settings = dict(sensor_shape=[16, 1, 128], acquisition=acquisition, sensor_scale=2., sensor_offset=0.)
        groups = sensor_features(arrays, depth, camera, settings)
        model = MaterialNetwork([g.shape for g in groups]).eval()
        with torch.inference_mode():
            expected = model(*(torch.from_numpy(g[None]) for g in groups))[0].numpy()
        meta = dict(model_type="neural_depth_material", format_version=1, settings=settings, camera=camera,
                    input_shapes=[list(g.shape) for g in groups], members=[{}], baseline_sha256="unused")
        with TemporaryDirectory() as temp:
            path = Path(temp)
            torch.save(dict(meta=meta, states=[model.state_dict()]), path/"model.pt")
            (path/"acquisition.json").write_text(json.dumps(acquisition))
            (path/"camera.json").write_text(json.dumps(camera))
            np.save(path/"depth.npy", depth)
            np.save(path/"sensor_samples.npy", arrays*2.)
            np.save(path/"empty_sensor_samples.npy", np.zeros_like(arrays))
            with self.assertRaisesRegex(ValueError, "original baseline checkpoint"):
                NeuralDepthMaterial(path/"model.pt", path/"camera.json")
            predictor = NeuralDepthMaterial(path/"model.pt")
            estimate = predictor.predict_files(path, path/"depth.npy", path/"camera.json")
            np.testing.assert_allclose(estimate, expected, rtol=1e-5, atol=1e-6)
            with self.assertRaisesRegex(ValueError, "Depth camera differs"):
                predictor.predict(wave=arrays, depth=depth, camera={})
            with self.assertRaisesRegex(ValueError, "one-shot"):
                predictor.predict(wave=arrays[:3], depth=depth, camera=camera)
