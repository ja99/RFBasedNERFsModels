import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from EchoReconstructor import (EchoReconstructor, affine_shape, coordinates, evaluate, local_training_targets,
                               object_moments, open_archive, predict, restore, trace_features, train)


class EchoReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_features_keep_amplitude_timing_and_channel_identity(self):
        wave = torch.randn(2, 3, 2, 100)
        original = trace_features(wave)
        self.assertTrue(torch.isfinite(trace_features(torch.zeros_like(wave))).all())
        torch.testing.assert_close(trace_features(2*wave)[..., :-1], original[..., :-1])
        torch.testing.assert_close(trace_features(2*wave)[..., -1], original[..., -1] + np.log(2))
        torch.testing.assert_close(trace_features(wave.flip(2)), original.flip(2))
        self.assertFalse(torch.equal(trace_features(wave.roll(10, -1)), original))

    def test_training_geometry_translation_orientation_and_surface_coordinates(self):
        shape = (17, 18, 19)
        xyz = coordinates(shape, "cpu")
        center = torch.tensor((7., 8., 9.))
        delta = xyz-center
        xy = (delta[:, 0] + delta[:, 1]) / (2**.5)
        perp = (delta[:, 0] - delta[:, 1]) / (2**.5)
        field = torch.stack((xy.abs()-4.8, perp.abs()-1.8, delta[:, 2].abs()-2.2), dim=1).amax(1)
        sdf = field.reshape(1, *shape)
        moved = sdf.roll(2, dims=1)
        moments = object_moments(torch.cat((sdf, moved)), xyz)
        torch.testing.assert_close(moments[1, :3] - moments[0, :3], torch.tensor((2., 0., 0.)))
        torch.testing.assert_close(moments[0, 3:], moments[1, 3:])
        transform = affine_shape(moments)
        torch.testing.assert_close(torch.linalg.det(transform), torch.ones(2))
        self.assertGreater(transform[0, 1, 0], .2)
        correction, weights = local_training_targets(torch.cat((sdf, moved)), moments, 7)
        # Interior local samples of the translated object represent the same geometry.
        mask = (weights[0] > 0) & (weights[1] > 0)
        torch.testing.assert_close(correction[0, mask], correction[1, mask], atol=2e-5, rtol=1e-5)

    def test_training_never_reads_test_and_prediction_needs_no_voxel_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dataset.hdf5"
            dims, rows, shots, times = (13, 12, 14), 4, 2, 36
            indices = np.moveaxis(np.indices(dims), 0, -1)
            signals, fields = [], []
            rng = np.random.default_rng(7)
            for i in range(16):
                center = rng.uniform([4, 4, 4], [8, 7, 9])
                radius = rng.uniform(2., 3.2)
                field = np.linalg.norm(indices-center, axis=-1)-radius
                grid = np.zeros((*dims, 3), dtype=np.float32)
                grid[..., 2] = field / 20 + .25
                fields.append(grid)
                receiver_y = np.linspace(2, 10, rows)[:, None, None]
                shot_z = np.linspace(3, 11, shots)[None, :, None]
                delay = np.sqrt(center[0]**2+(receiver_y-center[1])**2+(shot_z-center[2])**2)
                time = np.arange(times)[None,None,:]-delay*2
                signals.append((radius*np.cos(time*.7)*np.exp(-(time/3)**2)).astype(np.float32))
            with h5py.File(path, "w") as f:
                f["grids"] = fields
                f["sensor_samples"] = np.stack(signals).reshape(len(signals), rows*shots, times)
                f["scene_ids"] = [f"s{i}".encode() for i in range(len(signals))]
                f["splits/train"] = list(range(10))
                f["splits/validation"] = [10, 11, 12]
                f["splits/test"] = [13, 14, 15]
                f.attrs.update(sensor_shape=(rows, shots, times), sensor_storage="antenna_direction,time",
                               grid_min=[0., 0., -5.], grid_scale=[1., 1., 20.],
                               waveform_mode="calibrated", preprocessing_id="trained-on-s0-to-s9")
                f.create_dataset("training_protocol", data=json.dumps({"scene_ids": {"train": list(range(10))}}),
                                 dtype=h5py.string_dtype())
            args = SimpleNamespace(data=root, checkpoint=root/"echo.pt", device="cpu", queries=48, batch_size=4,
                                   epochs=3, val_every=1, patience=10, hidden=32, seed=11, ridge=.1, lr=.001)
            from CustomDataset import SensorDataset
            getitem = SensorDataset.__getitem__
            def without_test(dataset, index):
                self.assertLess(index, 13)
                return getitem(dataset, index)
            with patch.object(SensorDataset, "__getitem__", without_test), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            saved = torch.load(args.checkpoint, weights_only=True)
            self.assertIn("validation_scenes", saved["validation"])
            self.assertIn(saved["epoch"], range(args.epochs+1))
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(SimpleNamespace(**vars(args), ablate=True))
            self.assertEqual(set(report), {"measured", "zero_echo", "shuffled_echo"})
            self.assertEqual([r["scene_id"] for r in report["measured"]["scenes"]], ["s13", "s14", "s15"])
            # Removing every voxel label exercises the prediction interface on an actual signal-only archive.
            with h5py.File(path, "r+") as f:
                del f["grids"]
            result = root / "pred.npz"
            with contextlib.redirect_stdout(io.StringIO()):
                predict(SimpleNamespace(**vars(args), scene_id="s14", output=result))
            with np.load(result) as output:
                self.assertEqual(output["sdf_cells"].shape, dims)
                np.testing.assert_equal(output["occupied"], output["sdf_cells"] < 0)
                self.assertTrue(np.isfinite(output["sdf_cells"]).all())
            with h5py.File(path, "r+") as f:
                f.attrs["preprocessing_id"] = "other-study"
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                restore(args.checkpoint, open_archive(root, labels=False), "cpu")


if __name__ == "__main__":
    unittest.main()
