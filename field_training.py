"""Training and evaluation shared by the RF field reconstruction models."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset

from helpers import sample_indices, uniform_query_indices


DEFAULT_DATA_PATH = Path("new_beamforming_data")


def check_preprocessing(dataset, waveform_mode, preprocessing_id=None):
    if dataset is None:
        return
    actual = getattr(dataset, "waveform_mode", None)
    if waveform_mode is not None and actual is not None and waveform_mode != actual:
        raise ValueError(f"Checkpoint expects {waveform_mode} waveforms, dataset supplies {actual}")
    historical = waveform_mode == "legacy_raw" and preprocessing_id is None
    if historical and getattr(dataset, "training_protocol", None) is not None:
        raise ValueError("This raw checkpoint has no saved preprocessing transform and requires its original "
                         "historical training archive")
    if preprocessing_id is not None and getattr(dataset, "preprocessing_id", None) != preprocessing_id:
        raise ValueError("Checkpoint and dataset use different preprocessing. Convert evaluation scenes with "
                         "--normalization-from pointing to the training archive")


class SensorModel(pl.LightningModule):
    """Carry the input convention into checkpoints, including pre-calibration runs."""

    checkpoint_waveform_mode = None
    checkpoint_preprocessing_id = None
    checkpoint_training_protocol = None

    def _datasets(self, loader_name):
        datamodule = getattr(self._trainer, "datamodule", None)
        source = getattr(datamodule, "dataset", None)
        if source is not None:
            yield source
        loaders = getattr(self._trainer, loader_name, None)
        if isinstance(loaders, dict):
            loaders = loaders.values()
        elif not isinstance(loaders, (list, tuple)):
            loaders = (loaders,)
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            while isinstance(dataset, Subset):
                dataset = dataset.dataset
            if dataset is not source and hasattr(dataset, "waveform_mode"):
                yield dataset

    def on_load_checkpoint(self, checkpoint):
        self.checkpoint_waveform_mode = checkpoint.get("waveform_mode", "legacy_raw")
        self.checkpoint_preprocessing_id = checkpoint.get("preprocessing_id")
        self.checkpoint_training_protocol = checkpoint.get("training_protocol")
        for dataset in self._datasets("train_dataloader"):
            self.check_dataset(dataset)
        if "output_relu" in self.hparams:
            # Older voxel decoders and the voltage autoencoder ended in a ReLU.
            output_relu = checkpoint.get("hyper_parameters", {}).get("output_relu", True)
            self.hparams.output_relu = output_relu
            self.decoder[-1] = torch.nn.ReLU() if output_relu else torch.nn.Identity()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["waveform_mode"] = self.checkpoint_waveform_mode
        if self.checkpoint_preprocessing_id is not None:
            checkpoint["preprocessing_id"] = self.checkpoint_preprocessing_id
        if self.checkpoint_training_protocol is not None:
            checkpoint["training_protocol"] = self.checkpoint_training_protocol

    def check_dataset(self, dataset):
        check_preprocessing(dataset, self.checkpoint_waveform_mode, self.checkpoint_preprocessing_id)

    def on_fit_start(self):
        self._bind_training_data()

    def on_train_start(self):
        # Lightning attaches direct training DataLoaders after on_fit_start.
        self._bind_training_data()

    def _bind_training_data(self):
        for dataset in self._datasets("train_dataloader"):
            self.check_dataset(dataset)
            self.checkpoint_waveform_mode = getattr(dataset, "waveform_mode", self.checkpoint_waveform_mode)
            self.checkpoint_preprocessing_id = getattr(dataset, "preprocessing_id", self.checkpoint_preprocessing_id)
            self.checkpoint_training_protocol = getattr(dataset, "training_protocol", self.checkpoint_training_protocol)

    def on_validation_start(self):
        for dataset in self._datasets("val_dataloaders"):
            self.check_dataset(dataset)

    def on_test_start(self):
        for dataset in self._datasets("test_dataloaders"):
            self.check_dataset(dataset)

    def on_predict_start(self):
        for dataset in self._datasets("predict_dataloaders"):
            self.check_dataset(dataset)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PointQueryModel(SensorModel):
    """Use one set of indices for positions and labels in each scene.

    Training balances material or distance bins if requested by the architecture.
    Validation measures MSE on a fixed, label-independent sample of the volume.
    """

    target_channel = None
    stratify_training = True
    validation_step_counter = 0
    render_chunk_size = 2048
    native_stride = 1

    def encode_scene(self, voltages):
        return self.encoder(voltages.flatten(start_dim=2)).flatten(start_dim=1)

    def decode_queries(self, positions, scene):
        scene = scene.unsqueeze(1).expand(-1, positions.size(1), -1)
        return self.decoder(torch.cat((positions, scene), dim=-1))

    def predict_queries(self, positions, voltages):
        return self.decode_queries(positions, self.encode_scene(voltages))

    def predict_volume(self, batch, spatial_shape, chunk_size=None):
        positions, _, voltages = batch
        if chunk_size is None:
            chunk_size = self.render_chunk_size if self.device.type == "cuda" else min(512, self.render_chunk_size)
        scene = self.encode_scene(voltages.to(self.device))
        chunks = [self.decode_queries(part.to(self.device), scene).cpu() for part in positions.split(chunk_size, dim=1)]
        return torch.cat(chunks, dim=1).reshape(*spatial_shape, -1)

    def forward(self, batch, batch_idx=0):
        positions, _, voltages = batch
        # The historical rendering API returns [queries, channels] for one scene.
        return self.predict_queries(positions, voltages).squeeze(0)

    def _query_step(self, batch, batch_idx, *, validation):
        positions, grid, voltages = batch
        n_scenes, n_voxels = grid.shape[:2]
        if validation:
            indices = uniform_query_indices(n_scenes, n_voxels, self.internal_batch_size_validation,
                                            grid.device, seed=batch_idx)
        elif self.stratify_training:
            count = min(self.internal_batch_size_training, n_voxels)
            indices = torch.stack([sample_indices(scene, count, by_index=self.target_channel) for scene in grid])
        else:
            indices = uniform_query_indices(n_scenes, n_voxels, self.internal_batch_size_training, grid.device)

        scene_ids = torch.arange(n_scenes, device=grid.device)[:, None]
        targets = grid[scene_ids, indices]
        if self.target_channel is not None:
            targets = targets[..., self.target_channel:self.target_channel + 1]
        output = self.predict_queries(positions[scene_ids, indices], voltages)
        loss = self.loss_fn(output, targets)
        self.log("val_loss" if validation else "train_loss", loss, batch_size=n_scenes)
        if validation:
            self._log_histograms(targets, output)
        return loss

    def _log_histograms(self, targets, output):
        self.validation_step_counter += 1
        logger = self.logger
        if self.validation_step_counter < 3 or logger is None:
            return
        log = getattr(logger.experiment, "add_histogram", None)
        if log is not None:
            names = ("sdf",) if self.target_channel is not None else ("permittivity", "conductivity", "sdf")
            for i, name in enumerate(names[:output.size(-1)]):
                log(f"actual_values_{name}", targets[..., i], self.global_step)
                log(f"predicted_values_{name}", output[..., i], self.global_step)

    def training_step(self, batch, batch_idx):
        return self._query_step(batch, batch_idx, validation=False)

    def validation_step(self, batch, batch_idx):
        return self._query_step(batch, batch_idx, validation=True)


def evaluate_checkpoint(model_type, data_module, checkpoint_path, *, visualize=False, show_permittivity=True,
                        chunk_size=None, checkpoint_options=None):
    """Score every voxel of every test scene, or display the first test scene."""
    from Visualizer import sdf_metrics, visualize_permittivity, visualize_sdf_pair

    data_module.setup("test")
    if not len(data_module.test_dataset):
        raise ValueError("Evaluation requires a scene in the test split, but the split is empty")
    dataset = data_module.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_type.load_from_checkpoint(checkpoint_path, map_location="cpu", **(checkpoint_options or {}))
    model.to(device).eval()
    model.check_dataset(dataset)
    shape = dataset.grid_original_shape[:3]
    indices = iter(data_module.test_dataset.indices)
    scores_by_scene = {}
    with torch.inference_mode():
        for batch in data_module.test_dataloader():
            for row in range(batch[0].size(0)):
                scene = tuple(part[row:row + 1] for part in batch)
                index = next(indices)
                scene_id = dataset.scene_ids[index] if dataset.scene_ids is not None else f"hdf5-row-{index}"
                output = model.predict_volume(scene, shape, chunk_size)
                truth = scene[-2].reshape(*shape, -1)
                if visualize:
                    print(f"Test scene {scene_id}, grid {shape}, native prediction stride {model.native_stride}")
                    if show_permittivity:
                        visualize_permittivity(output[..., 0], "Predicted permittivity")
                        visualize_permittivity(truth[..., 0], "Ground truth permittivity")
                    return visualize_sdf_pair(output[..., -1], truth[..., -1], dataset)
                scores = sdf_metrics(output[..., -1], truth[..., -1], dataset.sdf_zero, dataset.sdf_scale)
                scores_by_scene[scene_id] = scores
                print(json.dumps({"scene_id": scene_id, **scores}))

    rows = list(scores_by_scene.values())
    summary = {"test_scenes": len(rows), "evaluation_grid_shape": list(shape), "native_stride": model.native_stride,
               "mean_per_scene": {
        key: sum(row[key] for row in rows) / len(rows) for key in rows[0] if not key.startswith("occupied_")
    }}
    if model.native_stride > 1:
        summary["prediction_resampling"] = "trilinear with constant extension after the final sampled cell per axis"
    if "occupied_union" in rows[0]:
        present = [row["occupancy_iou"] for row in rows if row["occupied_target"]]
        summary["empty_target_scenes"] = len(rows) - len(present)
        summary["mean_iou_nonempty_targets"] = sum(present) / len(present) if present else None
        for key in ("occupied_target", "occupied_predicted", "occupied_intersection", "occupied_union"):
            summary[key] = sum(row[key] for row in rows)
        union = summary["occupied_union"]
        summary["pooled_occupancy_iou"] = summary["occupied_intersection"] / union if union else 1.0
    print(json.dumps(summary))
    return {"scenes": scores_by_scene, "summary": summary}


def run_cli(train, render=None, *, pretrained_encoder=False, configure_train=None, configure_evaluation=None):
    parser = argparse.ArgumentParser(description="Train or evaluate an RF reconstruction model.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--seed", type=int, default=42, help="Initialize training and DataLoader RNGs")
    if pretrained_encoder:
        train_parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    if configure_train is not None:
        configure_train(train_parser)
    if render is not None:
        for command, description in (("render", "Visualize the first held-out scene"),
                                     ("evaluate", "Score all voxels in every held-out scene")):
            subparser = commands.add_parser(command, help=description)
            subparser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
            subparser.add_argument("--checkpoint", type=Path, required=True)
            if configure_evaluation is not None:
                configure_evaluation(subparser)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    options = vars(args).copy()
    command = options.pop("command")
    data = options.pop("data")
    if command in ("render", "evaluate"):
        return render(data, options.pop("checkpoint"), visualize=command == "render", **options)
    pl.seed_everything(options.pop("seed"), workers=True)
    return train(data, **options)
