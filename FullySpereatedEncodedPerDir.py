import datetime
import re
from copy import deepcopy

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from field_training import DEFAULT_DATA_PATH, PointQueryModel, evaluate_checkpoint, run_cli


class FullyEndToEnd(PointQueryModel):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16, n_directions: int = 5):
        super().__init__()

        self.encoders_per_dir = nn.ModuleList(
            nn.Sequential(nn.Linear(n_timesteps, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU())
            for _ in range(n_directions)
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_antennas * 512, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim_position + 8192, 16384),
            nn.ReLU(),
            nn.Linear(16384, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, dim_output),
        )

        self.lr = learning_rate
        self.internal_batch_size_training = internal_batch_size_training
        self.internal_batch_size_validation = internal_batch_size_validation
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def encode_scene(self, voltages):
        if voltages.size(2) != len(self.encoders_per_dir):
            raise ValueError("Recording direction count does not match this checkpoint")
        features = [encoder(voltages[:, :, i]) for i, encoder in enumerate(self.encoders_per_dir)]
        return self.encoder(torch.stack(features, dim=2).sum(dim=2).flatten(1))

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if "n_directions" in checkpoint.get("hyper_parameters", {}):
            return
        # Legacy state_dicts already contain a key for each direction, all with
        # identical values. Adam stored moments only for the single shared copy.
        # Map the archived single parameter group onto the now-independent copies.
        names = [name for name, _ in self.named_parameters()]
        old_names = [re.sub(r"^encoders_per_dir\.\d+\.", "encoders_per_dir.0.", name) for name in names]
        unique_old = list(dict.fromkeys(old_names))
        for opt in checkpoint.get("optimizer_states", ()):
            groups = opt["param_groups"]
            if len(groups) != 1 or len(groups[0]["params"]) != len(unique_old) or len(names) == len(unique_old):
                continue
            old_ids = dict(zip(unique_old, groups[0]["params"]))
            states = opt["state"]
            migrated = {}
            for i, (name, old_name) in enumerate(zip(names, old_names)):
                old_id = old_ids[old_name]
                if old_id in states:
                    migrated[i] = states[old_id] if name == old_name else deepcopy(states[old_id])
            opt["state"] = migrated
            groups[0]["params"] = list(range(len(names)))


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = FullyEndToEnd(
        n_antennas=n_antennas,
        n_directions=n_directions,
        n_timesteps=n_timesteps,
        dim_position=3,
        dim_output=3,
        learning_rate=1e-5,
        internal_batch_size_training=64,
        internal_batch_size_validation=64,
    )

    PATH = "FullySpereatedEncodedPerDir_logs"

    early_stop_callback = EarlyStopping(monitor=metric, min_delta=0.00, patience=25, verbose=True, mode="min")
    logger = TensorBoardLogger(save_dir=PATH, version=f"beamforming_{datetime.datetime.now()}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"{PATH}/{logger.version}", save_top_k=1,
                                          monitor=metric)

    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    logger.log_graph(model)

    trainer.fit(model, datamodule=data_module)


def render(data_path, checkpoint_path, *, visualize=True):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )
    return evaluate_checkpoint(FullyEndToEnd, data_module, checkpoint_path, visualize=visualize, show_permittivity=True)


if __name__ == "__main__":
    run_cli(normal_train, render)
