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

from EchoReconstructor import (VolumeReconstructor, affine_shape, coordinates, evaluate, local_training_targets,
                               object_moments, open_archive, predict, restore, score, trace_features, train)
from echo_components import component_counts


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

    def test_unrestricted_decoder_produces_disconnected_regions_on_original_lattice(self):
        shape = (23, 22, 21)
        feature_shape = (2, 2, 8)
        model = VolumeReconstructor(feature_shape, shape, hidden=16)
        features = torch.zeros(2, *feature_shape)
        native = model.spatial_mean.reshape(model.seed_shape)
        native.fill_(1.)
        native[1:3, 1:3, 1:3] = -1.
        native[-2:, -2:, -2:] = -1.
        field = model.encode(features)
        self.assertEqual(field.shape, (2, *shape))
        truth = field.detach()
        report = score(model, features, truth, coordinates(shape, "cpu"), ["a", "b"], 2)
        self.assertEqual(report["summary"]["matched_components_iou50"], 4)
        self.assertEqual(report["summary"]["by_target_components"]["2"]["scenes"], 2)
        self.assertEqual(report["summary"]["physical_instances"]["matched_components_iou50"], 4)
        self.assertEqual(report["summary"]["by_target_instances"]["2"]["physical_instances"]["count_accuracy"], 1.)
        self.assertEqual(report["summary"]["mean_scene_iou"], 1.)
        model.loss(features, truth - .2).backward()
        self.assertGreater(model.fine[-1].weight.grad.abs().sum(), 0.)

    def test_component_matching_counts_a_joined_pair_once(self):
        truth = np.zeros((7, 5, 5), dtype=bool)
        truth[0:2, 1:4, 1:4] = True
        truth[5:7, 1:4, 1:4] = True
        pred = truth.copy()
        pred[2:5, 2, 2] = True
        counts = component_counts(pred, truth)
        self.assertEqual(counts, dict(target_components=2, predicted_components=1, matched_components_iou25=1,
                                      matched_components_iou50=0))
        self.assertEqual(component_counts(np.zeros_like(pred), truth)["matched_components_iou25"], 0)

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
            acquisition = {"sensor_layout": "antenna,direction,timestep", "frequency_hz": 2.4e9}
            protocol = {"scene_ids": {"train": list(range(10))},
                        "settings": {"sensor_shape": [rows, shots, times], "sensor_storage": "antenna_direction,time",
                                     "acquisition": acquisition},
                        "normalization": {"sensor_offset": 0., "sensor_scale": .25}}
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
                f.create_dataset("training_protocol", data=json.dumps(protocol), dtype=h5py.string_dtype())
            args = SimpleNamespace(data=root, checkpoint=root/"echo.pt", device="cpu", queries=48, batch_size=4,
                                   epochs=3, val_every=1, patience=10, hidden=32, seed=11, ridge=.1, lr=.001)
            from CustomDataset import SensorDataset
            getitem = SensorDataset.__getitem__
            def without_test(dataset, index):
                self.assertLess(index, 13)
                return getitem(dataset, index)
            with patch.object(SensorDataset, "__getitem__", without_test), contextlib.redirect_stdout(io.StringIO()):
                train(args)
                mixed = SimpleNamespace(**{**vars(args), "checkpoint": root/"volume.pt", "decoder": "volume"})
                train(mixed)
            saved = torch.load(args.checkpoint, weights_only=True)
            self.assertIn("validation_scenes", saved["validation"])
            self.assertIn(saved["epoch"], range(args.epochs+1))
            with contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(SimpleNamespace(**vars(args), ablate=True))
                dense_report = evaluate(SimpleNamespace(**vars(mixed), ablate=False))
            self.assertEqual(dense_report["measured"]["summary"]["test_scenes"], 3)
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
                expected = output["sdf_cells"]
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "acquisition.json").write_text(json.dumps(acquisition))
            reference = np.full((rows, shots, times), 1e5)
            np.save(raw_dir / "empty_sensor_samples.npy", reference)
            np.save(raw_dir / "sensor_samples.npy", reference + .25 * signals[14].astype(np.float64))
            raw = SimpleNamespace(checkpoint=args.checkpoint, device="cpu", scene_dir=raw_dir,
                                  scene_id=None, output=root/"raw.npz")
            with contextlib.redirect_stdout(io.StringIO()):
                predict(raw)
            with np.load(raw.output) as output:
                np.testing.assert_allclose(output["sdf_cells"], expected, atol=2e-5)
                self.assertEqual(str(output["scene_id"]), "raw")
            (raw_dir/"acquisition.json").write_text(json.dumps({**acquisition, "frequency_hz": 1.}))
            with self.assertRaisesRegex(ValueError, "acquisition"):
                predict(raw)
            with h5py.File(path, "r+") as f:
                f.attrs["preprocessing_id"] = "other-study"
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                restore(args.checkpoint, open_archive(root, labels=False), "cpu")


if __name__ == "__main__":
    unittest.main()
