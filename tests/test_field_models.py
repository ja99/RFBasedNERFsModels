import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from field_training import evaluate_checkpoint
from FullyAutoEncoder import FullyEndToEnd as Autoencoder
from FullyEncodedPerAntenna import FullyEndToEnd as AntennaModel
from FullyEncodedPerTimestep import FullyEndToEnd as TimestepModel
from FullyINR import FullyNetwork
from FullyNonINR import FullyEndToEnd as NonINR
from FullySimplifiedEncodedPerAntenna import FullyEndToEnd as AntennaSDF
from FullySimplifiedNonINR import FullyEndToEnd as WholeGrid
from FullySimplifiedNonINR import render as whole_grid_render, upsample_sdf
from FullySimplifiedSpereatedEncodedPerDir import FullyEndToEnd as DirectionSDF
from FullySpereatedEncodedPerDir import FullyEndToEnd as DirectionModel
from helpers import sample_indices, uniform_query_indices
from TransformerPerAntennaAndDir import CustomTransformer
from TransformerPerAntennaAndDir import render as transformer_render


def small_direction_model(model_type=DirectionModel):
    with torch.device("meta"):
        model = model_type(2, 3, 3, 1, n_directions=2, internal_batch_size_validation=5)
    # Keep the two original direction encoders, replacing only the large downstream layers.
    model.encoders_per_dir.to_empty(device="cpu")
    for encoder in model.encoders_per_dir:
        for layer in encoder:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    model.encoder = nn.Linear(1024, 4)
    model.decoder = nn.Linear(7, 1)
    return model


def small_transformer(mode):
    return CustomTransformer(4, 3, 3, 3, 8, 2, 1, 1, 16, dropout=0.,
                             internal_batch_size_training=2, internal_batch_size_validation=5,
                             position_encoding=mode)


class FieldModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_stratification_uses_nonempty_bins_and_exhausts_unique_voxels(self):
        labels = torch.zeros(1000, 2)
        labels[-1, 1] = 0.95
        indices = sample_indices(labels, 64, by_index=1)
        self.assertEqual(len(indices.unique()), 64)
        self.assertEqual((indices == 999).sum().item(), 1)
        labels[500:, 0] = 1
        indices = sample_indices(labels, 64, by_index=0)
        self.assertEqual((labels[indices, 0] == 1).sum().item(), 32)
        self.assertEqual(sample_indices(labels[:3], 5).unique().numel(), 3)
        self.assertEqual(sample_indices(labels[:1], 1).item(), 0)

    def test_validation_queries_cover_volume_and_leave_training_rng_untouched(self):
        state = torch.random.get_rng_state()
        first = uniform_query_indices(3, 101, 10, "cpu", seed=8)
        second = uniform_query_indices(3, 101, 10, "cpu", seed=8)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertGreaterEqual(first[:, -1].min().item(), 90)
        full = uniform_query_indices(2, 9, 20, "cpu", seed=9)
        self.assertTrue(torch.equal(full, torch.arange(9).expand(2, -1)))

    def test_noninr_queries_gather_targets_from_each_entire_volume(self):
        with torch.device("meta"):
            model = NonINR(1, 2, 3, 3, n_directions=1,
                           internal_batch_size_training=9, internal_batch_size_validation=11)
        class Coordinates(nn.Module):
            def forward(self, x):
                return x[..., -3:]
        model.encoder = Coordinates()
        model.decoder = nn.Identity()
        model.log = lambda *args, **kwargs: None
        seen = []
        model.encoder.register_forward_hook(lambda _, __, out: seen.append(out[:, :, 0, 0]))
        ids = torch.arange(101).float().view(1, -1, 1).expand(2, -1, 3)
        ids = ids + torch.tensor([0., 1000.])[:, None, None]
        batch = ids, ids, torch.zeros(2, 1, 1, 2)
        self.assertEqual(model.training_step(batch, 0).item(), 0.)
        self.assertEqual(seen[-1].shape, (2, 9))
        self.assertGreaterEqual(seen[-1][0].max().item(), 89)
        self.assertGreaterEqual(seen[-1][1].max().item(), 1089)
        self.assertEqual(model.validation_step(batch, 0).item(), 0.)
        self.assertEqual(seen[-1].shape, (2, 11))
        singleton = ids[:1, :1], ids[:1, :1], batch[-1][:1]
        self.assertEqual(model.training_step(singleton, 0).item(), 0.)
        self.assertEqual(model(singleton).shape, (1, 3))

    def test_both_direction_models_preserve_shot_identity_and_gradients(self):
        waves = torch.randn(2, 2, 2, 3)
        for cls in (DirectionModel, DirectionSDF):
            model = small_direction_model(cls)
            self.assertIsNot(model.encoders_per_dir[0], model.encoders_per_dir[1])
            original = model.encode_scene(waves)
            self.assertFalse(torch.allclose(original, model.encode_scene(waves.flip(2))))
            original.square().mean().backward()
            for encoder in model.encoders_per_dir:
                self.assertGreater(encoder[0].weight.grad.abs().sum().item(), 0.)

    def test_archived_direction_weights_and_adam_moments_resume_independently(self):
        legacy = small_direction_model()
        legacy.encoders_per_dir[1] = legacy.encoders_per_dir[0]
        waves = torch.randn(1, 2, 2, 3)
        pos = torch.randn(1, 2, 3)
        opt = legacy.configure_optimizers()
        legacy.predict_queries(pos, waves).sum().backward()
        opt.step()
        old = {"optimizer_states": [opt.state_dict()]}
        fixed = small_direction_model()
        fixed.on_load_checkpoint(old)
        fixed.load_state_dict(legacy.state_dict())
        self.assertTrue(torch.equal(fixed.predict_queries(pos, waves), legacy.predict_queries(pos, waves)))
        resumed = fixed.configure_optimizers()
        resumed.load_state_dict(old["optimizer_states"][0])
        left = resumed.state[fixed.encoders_per_dir[0][0].weight]["exp_avg"]
        right = resumed.state[fixed.encoders_per_dir[1][0].weight]["exp_avg"]
        self.assertTrue(torch.equal(left, right))
        self.assertNotEqual(left.data_ptr(), right.data_ptr())
        resumed.zero_grad()
        fixed.predict_queries(pos, waves).sum().backward()
        resumed.step()
        self.assertFalse(torch.equal(fixed.encoders_per_dir[0][0].weight, fixed.encoders_per_dir[1][0].weight))

    def test_configured_dimensions_and_singleton_shape_for_dense_models(self):
        with torch.device("meta"):
            for cls in (AntennaModel, TimestepModel, AntennaSDF):
                model = cls(2, 7, 3, 1, n_directions=3)
                for batch, queries in ((1, 1), (2, 3)):
                    positions = torch.empty(batch, queries, 3)
                    waves = torch.empty(batch, 2, 3, 7)
                    self.assertEqual(model.predict_queries(positions, waves).shape, (batch, queries, 1))
            waves = torch.empty(2, 2, 3, 7)
            autoencoder = Autoencoder(2, 7, 3, 1, n_directions=3)
            self.assertEqual(autoencoder(waves).shape, (2, 42))
            self.assertIsInstance(autoencoder.decoder[-1], nn.Identity)
            autoencoder.on_load_checkpoint({})
            self.assertIsInstance(autoencoder.decoder[-1], nn.ReLU)
            self.assertTrue(autoencoder.hparams.output_relu)
            whole_grid = WholeGrid(2, 7, 3, 11, n_directions=3)
            self.assertEqual(whole_grid((None, waves)).shape, (2, 11))

    def test_transformer_chunking_and_legacy_checkpoint_encodings(self):
        waves, pos = torch.rand(2, 4, 3), torch.rand(2, 6, 3)
        for mode in ("sinusoidal", "learned"):
            model = small_transformer(mode).eval()
            with torch.no_grad():
                expected = model.predict_queries(pos, waves)
                chunked = torch.cat([model.predict_queries(q, waves) for q in pos.split(1, dim=1)], dim=1)
                split_input = model.predict_queries(pos, waves.reshape(2, 2, 2, 3))
                # Historical PyTorch decoder used length-one targets and repeated the memory for every voxel.
                decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(8, 2, 16, 0., batch_first=True), 1)
                decoder.load_state_dict(model.transformer_decoder.state_dict())
                memory = model.encode_scene(waves).repeat_interleave(6, dim=0)
                q = model.decoder_embedding(pos).reshape(12, 1, 8)
                legacy_output = model.output_layer(decoder.eval()(q, memory)).reshape_as(expected)
            torch.testing.assert_close(expected, chunked)
            torch.testing.assert_close(expected, split_input)
            torch.testing.assert_close(expected, legacy_output)
            hp = dict(model.hparams)
            del hp["position_encoding"]
            archived = {"pytorch-lightning_version": pl.__version__, "hyper_parameters": hp,
                        "state_dict": model.state_dict()}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "legacy.ckpt"
                torch.save(archived, path)
                restored = CustomTransformer.load_from_checkpoint(path).eval()
                self.assertEqual(restored.hparams.position_encoding, mode)
                torch.testing.assert_close(restored.predict_queries(pos, waves), expected)
                with self.assertRaisesRegex(ValueError, "legacy_raw"):
                    restored.check_dataset(SimpleNamespace(waveform_mode="calibrated"))

    def test_render_encodes_once_and_evaluation_scores_each_test_scene(self):
        positions = torch.arange(48).float().reshape(2, 8, 3)
        module = Mock()
        module.dataset = SimpleNamespace(grid_original_shape=(2, 2, 2, 3), waveform_mode="calibrated",
                                         sdf_zero=5., sdf_scale=3., scene_ids=["holdout-b", "holdout-a"])
        module.test_dataset = Subset(TensorDataset(positions), [1, 0])
        module.test_dataloader.return_value = [(positions, positions, torch.ones(2, 6))]
        model = FullyNetwork(2, 3, [8], 4, [8])
        with patch.object(model, "encode_scene", wraps=model.encode_scene) as encode, \
                patch.object(FullyNetwork, "load_from_checkpoint", return_value=model), \
                patch("Visualizer.visualize_sdf_pair") as draw:
            evaluate_checkpoint(FullyNetwork, module, "checkpoint", visualize=True,
                                show_permittivity=False, chunk_size=3)
            self.assertEqual(encode.call_count, 1)
            self.assertEqual(encode.call_args.args[0].shape, (1, 6))
            pred, truth, dataset = draw.call_args.args
            self.assertEqual(pred.shape, (2, 2, 2))
            torch.testing.assert_close(truth, positions[0, :, 2].reshape_as(truth))
            report = evaluate_checkpoint(FullyNetwork, module, "checkpoint", chunk_size=3)
            self.assertEqual(encode.call_count, 3)
            self.assertEqual(list(report["scenes"]), ["holdout-a", "holdout-b"])
            self.assertEqual(report["summary"]["test_scenes"], 2)
            self.assertIn("occupancy_iou", report["summary"]["mean_per_scene"])
            self.assertEqual(report["summary"]["occupied_target"], 1)
            self.assertEqual(report["summary"]["pooled_occupancy_iou"], 1 / 16)
        module.test_dataset = Subset(TensorDataset(positions), [])
        with self.assertRaisesRegex(ValueError, "test split"):
            evaluate_checkpoint(FullyNetwork, module, "checkpoint")

    def test_transformer_reads_single_shot_time_antenna_archive(self):
        samples = np.arange(6, dtype=np.float32).reshape(3, 2)
        model = CustomTransformer(2, 3, 3, 3, 8, 2, 1, 1, 16)
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp) / "dataset.hdf5", "w") as f:
                f["grids"] = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
                f["positions"] = np.indices((2, 2, 2)).reshape(3, -1).T
                f["sensor_samples"] = samples[None]
                for name, members in (("train", []), ("validation", []), ("test", [0])):
                    f[f"splits/{name}"] = np.array(members, dtype=np.int64)
            with patch.object(CustomTransformer, "load_from_checkpoint", return_value=model), \
                    patch.object(model, "encode_scene", wraps=model.encode_scene) as encode:
                scores = transformer_render(tmp, "checkpoint", visualize=False)
                canonical = torch.from_numpy(samples.T.copy()).reshape(1, 2, 1, 3)
                torch.testing.assert_close(encode.call_args.args[0].cpu(), canonical)
                self.assertEqual(list(scores["scenes"]), ["hdf5-row-0"])

    def test_historical_raw_checkpoint_requires_original_archive(self):
        legacy = SimpleNamespace(waveform_mode="legacy_raw", training_protocol=None)
        converted_raw = SimpleNamespace(waveform_mode="legacy_raw", preprocessing_id="v2",
                                        training_protocol={"splits": {}})
        model = FullyNetwork(2, 3)
        model.on_load_checkpoint({})
        model.check_dataset(legacy)
        with self.assertRaisesRegex(ValueError, "historical"):
            model.check_dataset(converted_raw)
        model.on_load_checkpoint({"waveform_mode": "legacy_raw", "preprocessing_id": "v2"})
        model.check_dataset(converted_raw)
        with self.assertRaisesRegex(ValueError, "normalization-from"):
            model.check_dataset(legacy)

    def test_archived_inr_infers_all_widths_and_accepts_activation_override(self):
        waves, points = torch.randn(2, 10), torch.randn(2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "archive.ckpt"
            for activation in (nn.GELU, nn.Tanh):
                original = FullyNetwork(2, 5, [7, 9], 6, [11, 4], activation=activation, output_activation=True)
                torch.save({"pytorch-lightning_version": pl.__version__, "state_dict": original.state_dict()}, path)
                if activation is nn.GELU:
                    with self.assertWarnsRegex(UserWarning, "GELU"):
                        restored = FullyNetwork.load_from_checkpoint(path)
                else:
                    restored = FullyNetwork.load_from_checkpoint(path, activation="Tanh", n_antennas=2)
                self.assertEqual(restored.hparams.n_antennas * restored.hparams.n_timesteps, 10)
                self.assertEqual(restored.hparams.encoder_dims_hidden, [7, 9])
                self.assertEqual(restored.hparams.position_dim_hidden, 6)
                self.assertEqual(restored.hparams.own_dims_hidden, [11, 4])
                torch.testing.assert_close(original(waves, points), restored(waves, points))

    def test_coarse_baseline_uses_true_stride_and_full_resolution_target(self):
        shape = (7, 9, 4)
        x, y, z = torch.meshgrid(*(torch.arange(n).float() for n in shape), indexing="ij")
        values = 100 * x + 10 * y + z
        coarse = values[::5, ::5, ::5]
        full = upsample_sdf(coarse, shape, 5)[..., 0]
        torch.testing.assert_close(full[::5, ::5, ::5], coarse)
        torch.testing.assert_close(full, 100 * x.clamp(max=5) + 10 * y.clamp(max=5))
        with tempfile.TemporaryDirectory() as tmp:
            shape = (23, 26, 19)
            grid = np.ones((*shape, 3), dtype=np.float32)
            # A five-cell cube rotated 45 degrees about z contains 125 occupied
            # centers on the full lattice. The stride-5 decoder sees only air.
            x, y, z = np.indices(shape) - np.array([12, 12, 10])[:, None, None, None]
            inside = ((np.abs(x + y) < 2.5 * np.sqrt(2)) & (np.abs(x - y) < 2.5 * np.sqrt(2))
                      & (np.abs(z) < 2.5))
            self.assertEqual(inside.sum(), 125)
            self.assertFalse(inside[::5, ::5, ::5].any())
            grid[..., 2][inside] = 0
            coarse_size = grid[::5, ::5, ::5, 2].size
            with h5py.File(Path(tmp) / "dataset.hdf5", "w") as f:
                f["grids"] = grid[None]
                f["subsampled_grids"] = grid[None, ::5, ::5, ::5]
                f["sensor_samples"] = np.ones((1, 3, 2), dtype=np.float32)
                f.attrs["grid_min"] = [0., 0., -0.5]
                f.attrs["grid_scale"] = [1., 1., 1.]
                for name, members in (("train", []), ("validation", []), ("test", [0])):
                    f[f"splits/{name}"] = np.array(members, dtype=np.int64)
            with torch.device("meta"):
                model = WholeGrid(2, 3, 3, coarse_size, n_directions=1)
            model.encoder = nn.Identity()
            model.decoder = nn.Linear(6, coarse_size)
            with torch.no_grad():
                model.decoder.weight.zero_()
                model.decoder.bias.fill_(1)
            with patch.object(WholeGrid, "load_from_checkpoint", return_value=model):
                report = whole_grid_render(tmp, "checkpoint", visualize=False)["summary"]
        self.assertEqual(report["evaluation_grid_shape"], list(shape))
        self.assertEqual(report["native_stride"], 5)
        self.assertEqual(report["occupied_target"], 125)
        self.assertEqual(report["empty_target_scenes"], 0)
        self.assertEqual(report["mean_iou_nonempty_targets"], 0.)
        self.assertEqual(report["pooled_occupancy_iou"], 0.)

    def test_actual_fit_and_checkpoint_roundtrip(self):
        positions = torch.rand(4, 16, 3)
        data = TensorDataset(positions, positions.clone(), torch.rand(4, 6))
        data.waveform_mode = "calibrated"
        data.preprocessing_id = "training-transform"
        data.training_protocol = {"splits": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}}
        class SceneModule(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
                self.dataset = data
            def train_dataloader(self):
                return DataLoader(data, batch_size=2)
            def val_dataloader(self):
                return DataLoader(data, batch_size=2)
        model = FullyNetwork(2, 3, [8], 4, [8], internal_batch_size=3, internal_batch_size_validation=7)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, default_root_dir=tmp,
                                 enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, datamodule=SceneModule())
            self.assertIn("val_loss", trainer.callback_metrics)
            path = Path(tmp) / "field.ckpt"
            trainer.save_checkpoint(path)
            restored = FullyNetwork.load_from_checkpoint(path)
            self.assertEqual(restored.checkpoint_waveform_mode, "calibrated")
            self.assertEqual(restored.checkpoint_training_protocol, data.training_protocol)
            self.assertEqual(restored.checkpoint_preprocessing_id, data.preprocessing_id)
            restored.check_dataset(data)
            with self.assertRaisesRegex(ValueError, "normalization-from"):
                restored.check_dataset(SimpleNamespace(waveform_mode="calibrated", preprocessing_id="other"))
            torch.testing.assert_close(restored.predict_queries(positions, data.tensors[-1]),
                                       model.predict_queries(positions, data.tensors[-1]))

    def test_direct_dataloaders_bind_and_check_preprocessing(self):
        points = torch.rand(3, 8, 3)
        data = TensorDataset(points, points, torch.rand(3, 6))
        data.waveform_mode = "calibrated"
        data.preprocessing_id = "train"
        model = FullyNetwork(2, 3, [8], 4, [8], internal_batch_size=3)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, default_root_dir=tmp,
                                 enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, train_dataloaders=DataLoader(Subset(data, [0, 1])),
                        val_dataloaders=DataLoader(Subset(data, [2])))
            self.assertEqual(model.checkpoint_preprocessing_id, "train")
            other = TensorDataset(*data.tensors)
            other.waveform_mode = "calibrated"
            other.preprocessing_id = "other"
            with self.assertRaisesRegex(ValueError, "normalization-from"):
                trainer.predict(model, dataloaders=DataLoader(Subset(other, [1])))


if __name__ == "__main__":
    unittest.main()
