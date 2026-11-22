import datetime
import math

from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset
from field_training import DEFAULT_DATA_PATH, SensorModel, evaluate_checkpoint, run_cli


def upsample_sdf(coarse, spatial_shape, stride):
    """Interpolate at original voxel coordinates and extend the last coarse value to each outer boundary."""
    coarse_shape = tuple((size - 1) // stride + 1 for size in spatial_shape)
    extent = tuple((size - 1) * stride + 1 for size in coarse_shape)
    values = coarse.reshape(1, 1, *coarse_shape)
    sampled = F.interpolate(values, size=extent, mode="trilinear", align_corners=True)
    tail_x, tail_y, tail_z = (size - sampled_size for size, sampled_size in zip(spatial_shape, extent))
    full = F.pad(sampled, (0, tail_z, 0, tail_y, 0, tail_x), mode="replicate")
    return full[0, 0].unsqueeze(-1)


class FullyEndToEnd(SensorModel):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16, n_directions: int = 5, output_relu: bool = False,
                 native_stride: int = 5):
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
            nn.Linear(8192, 16384),
            nn.ReLU(),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            nn.Linear(16384, dim_output),
            nn.ReLU() if output_relu else nn.Identity(),
        )

        self.lr = learning_rate
        self.native_stride = native_stride
        self.internal_batch_size_training = internal_batch_size_training
        self.internal_batch_size_validation = internal_batch_size_validation
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, batch, batch_idx=0):
        _, voltages = batch
        latent = self.encoder(voltages.flatten(start_dim=2)).flatten(start_dim=1)
        return self.decoder(latent)

    def predict_volume(self, batch, spatial_shape, chunk_size=None):
        _, voltages = batch
        coarse = self((None, voltages.to(self.device))).cpu()
        return upsample_sdf(coarse, spatial_shape, self.native_stride)

    def training_step(self, batch, batch_idx):
        loss = self.loss_fn(self(batch), batch[0][..., 2])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch[0][..., 2]
        output = self(batch)
        loss = self.loss_fn(output, target)
        log = getattr(getattr(self.logger, "experiment", None), "add_histogram", None)
        if batch_idx == 0 and log is not None:
            log("actual_values_sdf", target, self.global_step)
            log("predicted_values_sdf", output, self.global_step)
        self.log("val_loss", loss)
        return loss


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=64,
        load_positions=False,
        load_normal_grids=False,
        load_subsampled_grids=True,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    output_size = math.prod(data_module.dataset.grid_original_shape[:3])

    metric = data_module.checkpoint_metric()
    n_antennas, n_directions, n_timesteps = data_module.dataset.sensor_shape
    model = FullyEndToEnd(
        n_antennas=n_antennas,
        n_directions=n_directions,
        n_timesteps=n_timesteps,
        dim_position=3,
        dim_output=output_size,
        native_stride=data_module.dataset.subsample_stride,
        learning_rate=1e-5,
    )

    PATH = "FullySimplifiedNonINR_logs"

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
        load_positions=False,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )
    return evaluate_checkpoint(FullyEndToEnd, data_module, checkpoint_path, visualize=visualize,
                               show_permittivity=False)


if __name__ == "__main__":
    run_cli(normal_train, render)
