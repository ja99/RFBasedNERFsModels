import datetime

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from field_training import DEFAULT_DATA_PATH, SensorModel, run_cli


class FullyEndToEnd(SensorModel):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16, n_directions: int = 5, output_relu: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_directions * n_timesteps, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_antennas * 512, 8192),
            nn.ReLU(),
            nn.Linear(8192, n_antennas * n_directions * n_timesteps),
            nn.ReLU() if output_relu else nn.Identity(),
        )

        self.lr = learning_rate
        self.internal_batch_size_training = internal_batch_size_training
        self.internal_batch_size_validation = internal_batch_size_validation
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, voltages, batch_idx=0):
        if isinstance(voltages, (tuple, list)):
            voltages = voltages[-1]
        latent = self.encoder(voltages.flatten(start_dim=2)).flatten(start_dim=1)
        return self.decoder(latent)

    def training_step(self, voltages, batch_idx):
        loss = self.loss_fn(self(voltages), voltages.flatten(start_dim=1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, voltages, batch_idx):
        target = voltages.flatten(start_dim=1)
        output = self(voltages)
        loss = self.loss_fn(output, target)
        log = getattr(getattr(self.logger, "experiment", None), "add_histogram", None)
        if batch_idx == 0 and log is not None:
            log("actual_values", target, self.global_step)
            log("predicted_values", output, self.global_step)
        self.log("val_loss", loss)
        return loss


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=64,
        load_positions=False,
        load_normal_grids=False,
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
        learning_rate=1e-3,
        internal_batch_size_training=64,
        internal_batch_size_validation=64,
    )

    PATH = "FullyAutoencoder_logs"

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


if __name__ == "__main__":
    run_cli(normal_train)
