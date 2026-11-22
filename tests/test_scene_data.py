import json
from pathlib import Path
import pickle
import shutil
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
import torch

from CustomDataset import SensorDataModule, SensorDataset
from SaveAsHDF5 import save_to_hdf5, save_to_hdf5_beamforming, split_scene_indices
from Visualizer import sdf_metrics


class SceneDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shape = (6, 8, 7)
        self.trace_shape = (2, 5, 8)
        self.splits = split_scene_indices(5)
        self.echo = np.arange(np.prod(self.trace_shape)).reshape(self.trace_shape) - 40.
        # Sub-unit echoes on a large common waveform disappear if cast to float32 before subtraction.
        self.empty = np.full(self.trace_shape, 1e8)
        self.held = self.splits["test"][0]
        self.sdf = np.indices(self.shape)[0] - 2.
        for i in reversed(range(5)):
            scene = self.root / f"scene-{i}"
            scene.mkdir()
            grid = np.zeros((*self.shape, 3))
            grid[..., 0] = np.where(self.sdf < 0, 4 + i, 1)
            grid[..., 2] = self.sdf * (10 if i == self.held else 1)
            np.save(scene / "grid.npy", grid)
            np.save(scene / "sensor_samples.npy", self.empty + self.echo * self.factor(i))
            np.save(scene / "empty_sensor_samples.npy", self.empty)
            (scene / "acquisition.json").write_text(json.dumps({
                "sensor_layout": "antenna,direction,timestep", "samples_per_shot": 8,
                "beamforming_angles_deg": [[0, 0], [-30, 0], [30, 0], [0, -30], [0, 30]],
            }))

    def factor(self, i):
        return (1000 if i == self.held else i + 1) * .001

    def test_train_only_calibration_split_and_layout(self):
        output = save_to_hdf5_beamforming(self.root)
        dm = SensorDataModule(self.root, load_subsampled_grids=False, load_occupation_grids=True,
                              load_positions=True, flatten_sensor_samples=False, split_sensor_samples=True)
        ds = dm.dataset
        self.assertEqual(ds.sensor_shape, self.trace_shape)
        self.assertEqual(ds.waveform_mode, "calibrated")
        self.assertEqual(ds.scene_ids, [f"scene-{i}" for i in range(5)])
        self.assertAlmostEqual(ds.sdf_zero, .4)
        self.assertEqual(ds.sdf_scale, 5)
        dm.setup("fit")
        dm.setup("test")
        self.assertEqual(dm.checkpoint_metric(), "val_loss")
        self.assertEqual(dm.train_dataset.indices, self.splits["train"].tolist())
        self.assertEqual(dm.test_dataloader().dataset.indices, [self.held])
        self.assertEqual(dm.val_dataset.indices, self.splits["validation"].tolist())
        with h5py.File(output) as f:
            scale = max(self.factor(i) for i in self.splits["train"]) * np.max(abs(self.echo))
            self.assertAlmostEqual(f.attrs["sensor_scale"], scale)
            self.assertEqual(f["grids"].shape, (5, *self.shape, 3))
            for i in range(5):
                positions, labels, volts = ds[i]
                self.assertEqual(tuple(volts.shape), self.trace_shape)
                np.testing.assert_allclose(volts, self.echo * self.factor(i) / scale, atol=2e-5, rtol=1e-6)
                np.testing.assert_array_equal(labels[:, 0], (self.sdf < 0).ravel())
                self.assertEqual(tuple(positions.shape), (np.prod(self.shape), 3))
            self.assertGreater(f["sensor_samples"][self.held].max(), 1)
            self.assertGreater(f["grids"][self.held, ..., 2].max(), 1)
            self.assertTrue(np.isfinite(f["grids"][:]).all())  # conductivity is constant in training
        restored = pickle.loads(pickle.dumps(ds))
        torch.testing.assert_close(restored[0][2], ds[0][2])
        small = SensorDataset(self.root, load_positions=True)
        positions, grid, _ = small[0]
        np.testing.assert_array_equal(positions, ds.positions.reshape(*self.shape, 3)[::5, ::5, ::5].reshape(-1, 3))
        self.assertEqual(positions.shape[0], grid.shape[0])

    def test_failed_conversion_preserves_previous_archive(self):
        output = save_to_hdf5_beamforming(self.root)
        previous = output.read_bytes()
        path = self.root / "scene-4" / "grid.npy"
        np.save(path, np.zeros((3, 3, 3, 3)))
        with self.assertRaisesRegex(ValueError, "inconsistent grid"):
            save_to_hdf5_beamforming(self.root)
        self.assertEqual(output.read_bytes(), previous)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete scene scene-4"):
            save_to_hdf5_beamforming(self.root)

    def test_raw_requires_explicit_choice_and_shared_reference_works(self):
        for path in self.root.glob("*/empty_sensor_samples.npy"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "matched empty"):
            save_to_hdf5_beamforming(self.root)
        ref = self.root / "empty.npy"
        np.save(ref, self.empty)
        root_meta = self.root / "acquisition.json"
        root_meta.write_text(json.dumps({"frequency_hz": 1}))
        with self.assertRaisesRegex(ValueError, "matching acquisition"):
            save_to_hdf5_beamforming(self.root, reference_path=ref)
        shutil.copyfile(self.root / "scene-0" / "acquisition.json", root_meta)
        save_to_hdf5_beamforming(self.root, reference_path=ref)
        self.assertEqual(SensorDataset(self.root).waveform_mode, "calibrated")
        save_to_hdf5_beamforming(self.root, raw=True, crop_x_end=1)
        with h5py.File(self.root / "dataset.hdf5") as f:
            self.assertEqual(f["grids"].shape[1], self.shape[0] - 1)
            self.assertEqual(f.attrs["waveform_mode"], "legacy_raw")

    def test_legacy_archive_keeps_values_and_marks_missing_zero(self):
        with h5py.File(self.root / "dataset.hdf5", "w") as f:
            f["grids"] = np.zeros((5, 2, 2, 2, 3), np.float32)
            f["sensor_samples"] = np.arange(5 * 80 * 7, dtype=np.float32).reshape(5, 80, 7)
        kwargs = dict(load_normal_grids=True, load_subsampled_grids=False, split_sensor_samples=True,
                      flatten_sensor_samples=False)
        first, second = (SensorDataModule(self.root, **kwargs) for _ in range(2))
        with self.assertWarnsRegex(UserWarning, "historical"):
            first.setup()
        with self.assertWarns(UserWarning):
            second.setup()
        self.assertEqual(first.test_dataset.indices, second.test_dataset.indices)
        self.assertFalse(set(first.test_dataset.indices) & set(first.val_dataset.indices))
        self.assertIsNone(first.dataset.sdf_zero)
        grid, voltage = first.dataset[0]
        self.assertEqual(tuple(voltage.shape), (16, 5, 7))
        self.assertEqual(voltage[1, 2, 3], (1 * 5 + 2) * 7 + 3)
        self.assertEqual(sdf_metrics(grid[..., 2], grid[..., 2]), {"normalized_mae": 0.0})

    def test_legacy_packing_and_single_shot_axis_order(self):
        scene = self.root / "scene-0"
        for path in self.root.glob("scene-*/acquisition.json"):
            path.unlink()
        traces = np.arange(16 * 5 * 4).reshape(16, 5, 4)
        for directory in self.root.glob("scene-*"):
            np.save(directory / "sensor_samples.npy", traces.reshape(16, -1))
            np.save(directory / "empty_sensor_samples.npy", np.zeros((80, 4)))
        with self.assertRaisesRegex(ValueError, "ambiguous 2-D"):
            save_to_hdf5_beamforming(self.root)
        save_to_hdf5_beamforming(self.root, sensor_layout="packed-antenna-direction")
        ds = SensorDataset(self.root, load_subsampled_grids=False, split_sensor_samples=True,
                           flatten_sensor_samples=False)
        np.testing.assert_allclose(ds[0], traces / traces.max(), rtol=1e-6)
        for directory in self.root.glob("scene-*"):
            np.save(directory / "sensor_samples.npy", traces[:, 0, :])
        save_to_hdf5(self.root)
        single = SensorDataset(self.root, load_subsampled_grids=False, flatten_sensor_samples=False)
        raw = np.load(scene / "sensor_samples.npy")
        np.testing.assert_allclose(single[0], (raw / raw.max()).T)
        self.assertEqual(single.sensor_shape, (16, 1, 4))

    def test_new_test_scenes_reuse_the_fitted_archive(self):
        training = save_to_hdf5_beamforming(self.root)
        original = SensorDataset(self.root)
        external = self.root / "evaluation"
        external.mkdir()
        shutil.copytree(self.root / f"scene-{self.held}", external / "new-object")
        save_to_hdf5_beamforming(external, normalization_from=training)
        new = SensorDataset(external)
        self.assertEqual(new.splits, dict(train=[], validation=[], test=[0]))
        with self.assertRaisesRegex(ValueError, "no training scenes"):
            SensorDataModule(external).checkpoint_metric()
        self.assertEqual(new.preprocessing_id, original.preprocessing_id)
        self.assertEqual(new.training_protocol, original.training_protocol)
        for target, source in zip(new[0], original[self.held]):
            torch.testing.assert_close(target, source)
        train_id = original.training_protocol["scene_ids"]["train"][0]
        shutil.copytree(self.root / train_id, external / "renamed-training-scene")
        with self.assertRaisesRegex(ValueError, "overlap"):
            save_to_hdf5_beamforming(external, normalization_from=training)
        save_to_hdf5_beamforming(self.root, seed=43)
        self.assertNotEqual(SensorDataset(self.root).preprocessing_id, original.preprocessing_id)

    def test_near_zero_raw_occupancy_survives_normalization(self):
        scene = self.root / "scene-1" / "grid.npy"
        grid = np.load(scene)
        grid[2, 0, 0, 2] = -1e-12
        np.save(scene, grid)
        save_to_hdf5_beamforming(self.root)
        ds = SensorDataset(self.root, load_occupation_grids=True, flatten_grids=False, load_sensor_samples=False)
        labels = ds[1]
        self.assertEqual(labels[2, 0, 0, 0], 1)
        self.assertLess(labels[2, 0, 0, 1], ds.sdf_zero)
        self.assertEqual(labels[2, 0, 1, 0], 0)
        self.assertEqual(labels[2, 0, 1, 1], ds.sdf_zero)

    def test_renamed_duplicate_material_cannot_cross_splits(self):
        original_name = "scene-0"
        duplicate_name = "another-name"
        shutil.copytree(self.root / original_name, self.root / duplicate_name)
        save_to_hdf5_beamforming(self.root)
        dataset = SensorDataset(self.root)
        membership = dataset.training_protocol["scene_ids"]
        containing = [names for names in membership.values() if original_name in names]
        self.assertEqual(len(containing), 1)
        self.assertIn(duplicate_name, containing[0])
        singleton = self.root / "singleton"
        singleton.mkdir()
        shutil.copytree(self.root / original_name, singleton / original_name)
        save_to_hdf5_beamforming(singleton)
        with self.assertWarnsRegex(UserWarning, "training loss"):
            self.assertEqual(SensorDataModule(singleton).checkpoint_metric(), "train_loss")

    def test_similarity_uses_distinct_pairs_from_one_split(self):
        import pytorch_lightning as pl
        from SimilarityEncoder import EncoderModule, SensorDataModule as Pairs
        from SimilarityEncoder import SensorDataset as PairDataset

        save_to_hdf5_beamforming(self.root)
        pairs = Pairs(self.root)
        with self.assertWarnsRegex(UserWarning, "distinct scenes"):
            pairs.setup()
        self.assertEqual(len(pairs.val_dataset), 0)
        self.assertEqual(len(pairs.test_dataset), 0)
        available = [pairs.dataset[i][-1] for i in pairs.train_dataset.scenes.indices]
        for i in range(len(available)):
            pair = pairs.train_dataset[i]
            torch.testing.assert_close(pair["sensor_samples_a"], available[i])
            self.assertTrue(any(torch.equal(pair["sensor_samples_b"], other) for j, other in enumerate(available)
                                if i != j))
        evaluation_pairs = PairDataset(pairs.train_dataset.scenes, deterministic=True)
        torch.testing.assert_close(evaluation_pairs[0]["sensor_samples_b"], available[1])
        model = EncoderModule(sensor_samples_size=8, dims_hidden=[10, 3], activation=torch.nn.GELU)
        checkpoint = self.root / "similarity.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                    "pytorch-lightning_version": pl.__version__}, checkpoint)
        restored = EncoderModule.load_from_checkpoint(checkpoint)
        x = torch.arange(8, dtype=torch.float32)
        torch.testing.assert_close(model(x), restored(x))

    def test_physical_occupancy_metrics(self):
        truth = torch.tensor([0.2, 0.3, 0.4, 0.6])
        scores = sdf_metrics(torch.tensor([0.2, 0.5, 0.4, 0.7]), truth, sdf_zero=0.4, sdf_scale=10)
        self.assertEqual(scores["occupancy_iou"], .5)
        self.assertAlmostEqual(scores["sdf_mae_cells"], .75, places=5)
        self.assertEqual(sdf_metrics(truth, truth, sdf_zero=0)["occupancy_iou"], 1)

    def test_waveform_template_baseline_can_only_retrieve_training_scenes(self):
        from TemplateBaseline import evaluate_templates
        known = int(self.splits["train"][0])
        for i in range(5):
            echo = np.zeros(self.trace_shape)
            echo.flat[i] = 1.
            if i == self.held:
                echo.flat[known] = 2.
            np.save(self.root / f"scene-{i}" / "sensor_samples.npy", self.empty + echo)
        save_to_hdf5_beamforming(self.root)
        result = evaluate_templates(self.root, [self.root])
        self.assertEqual(result["summary"]["test_scenes"], 1)
        self.assertEqual(result["scenes"][0]["scene"], f"scene-{self.held}")
        self.assertEqual(result["scenes"][0]["template"], f"scene-{known}")


if __name__ == "__main__":
    unittest.main()
