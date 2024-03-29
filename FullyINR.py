import pathlib
import datetime
from typing import Tuple, Any, Dict, List

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from helpers import sample_indices

import CustomDataset
from helpers import *
import pytorch_lightning as pl
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F

# Define the LightningModule
class AntennaEncoderModule(nn.Module):
    def __init__(self,
                 sensor_samples_size: int = 1000,  # n_antenna*n_sample
                 dims_hidden: List[int] = [256],
                 activation: nn.Module = None,
                 ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(sensor_samples_size, dims_hidden[0]))
        layers.append(activation())
        for i in range(1, len(dims_hidden)):
            layers.append(nn.Linear(dims_hidden[i - 1], dims_hidden[i]))
            layers.append(activation())

        self.encode = nn.Sequential(*layers)

    def forward(self,
                inputs: torch.Tensor,  # [B, n_antennas*n_samples]
                ) -> torch.Tensor:  # [B, dims_hidden[-1]]
        latent = self.encode(inputs)
        return latent


class FullyNetwork(pl.LightningModule):
    def __init__(self,
                 n_antennas: int,
                 n_timesteps: int,
                 encoder_dims_hidden: List[int] = [256],
                 position_dim_hidden: int = 256,
                 own_dims_hidden: List[int] = [256],
                 activation: nn.Module = None,
                 learning_rate: float = 1e-6,
                 internal_batch_size: int = 100,
                 ):
        super().__init__()

        self.antenna_encoder = AntennaEncoderModule(
            sensor_samples_size=n_antennas * n_timesteps,
            dims_hidden=encoder_dims_hidden,
            activation=activation
        )





        self.position_encoder = nn.Sequential(
            nn.Linear(3, position_dim_hidden),
            activation(),
        )

        # self.position_encoder = nn.Identity()
        # position_dim_hidden = 3

        layers = []
        layers.append(nn.Linear(encoder_dims_hidden[-1] + position_dim_hidden, own_dims_hidden[0]))
        layers.append(activation())
        for i in range(1, len(own_dims_hidden)):
            layers.append(nn.Linear(own_dims_hidden[i - 1], own_dims_hidden[i]))
            layers.append(activation())

        layers.append(nn.Linear(own_dims_hidden[-1], 3))
        layers.append(activation())

        self.decode = nn.Sequential(*layers)

        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        self.internal_batch_size = internal_batch_size
        self.validation_step_counter = 0

    def forward(self,
                inputs: torch.Tensor,  # [B, n_antennas*n_samples]
                position: torch.Tensor,  # [B, 3]
                ) -> torch.Tensor:
        antennas_latent = self.antenna_encoder(inputs)
        position_latent = self.position_encoder(position)
        antennas_latent = torch.cat((antennas_latent, position_latent), dim=1)
        return self.decode(antennas_latent)

    def training_step(self,
                      inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      # [B, *grid_size, 3], [B, *grid_size*3], [B, n_antennas*n_samples]
                      batch_idx: int
                      ):
        positions, grid_values, voltages = inputs

        # Generate a set of random indices
        random_indices = sample_indices(grid_values, self.internal_batch_size)

        # Use the same indices to sample from both arrays
        positions = positions[0, random_indices]
        grid_values = grid_values[0, random_indices]

        positions_latents = self.position_encoder(positions).squeeze()
        antennas_latents = self.antenna_encoder(voltages)

        total_loss = 0
        for i in range(positions_latents.shape[0]):
            b = torch.cat((antennas_latents, positions_latents[i].unsqueeze(0)), dim=-1)
            decoded = self.decode(b).squeeze()
            loss = self.loss_fn(decoded, grid_values[i])
            total_loss += loss

        avg_loss = total_loss / positions_latents.shape[0]
        self.log('train_loss', avg_loss)
        return avg_loss

        # self.log('train_loss', total_loss)
        # return total_loss

    def validation_step(self,
                        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        # [B, *grid_size, 3], [B, *grid_size*3], [B, n_antennas*n_samples]
                        batch_idx: int,
                        ):
        positions, grid_values, voltages = inputs

        # Generate a set of random indices
        random_indices = sample_indices(grid_values, self.internal_batch_size)

        # Use the same indices to sample from both arrays
        positions = positions[0, random_indices]
        grid_values = grid_values[0, random_indices]

        positions_latents = self.position_encoder(positions).squeeze()
        antennas_latents = self.antenna_encoder(voltages)

        predictions = []

        total_loss = 0
        for i in range(positions_latents.shape[0]):
            b = torch.cat((antennas_latents, positions_latents[i].unsqueeze(0)), dim=-1)
            decoded = self.decode(b).squeeze()
            predictions.append(decoded)
            loss = self.loss_fn(decoded, grid_values[i])
            total_loss += loss

        if self.validation_step_counter >= 2:
            # draw histogram of predicted values and grid values for evaluation
            self.logger.experiment.add_histogram('actual_values_permittivity', grid_values[:, 0], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_permittivity', decoded[:, 0], self.global_step)
            self.logger.experiment.add_histogram('actual_values_conductivity', grid_values[:, 1], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_conductivity', decoded[:, 1], self.global_step)
            self.logger.experiment.add_histogram('actual_values_sdf', grid_values[:, 2], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_sdf', decoded[:, 2], self.global_step)

        self.validation_step_counter += 1

        avg_loss = total_loss / positions_latents.shape[0]
        self.log('val_loss', avg_loss)
        return avg_loss

        # self.log('val_loss', total_loss)
        # return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def normal_train():
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_sensor_samples=True,
        load_subsampled_grids=False,
        flatten_sensor_samples=True
    )
    model = FullyNetwork(
        n_antennas=16*5,
        n_timesteps=750,
        encoder_dims_hidden=[16384, 8192, 4096, 2048],
        position_dim_hidden=256,
        own_dims_hidden=[2048, 1024, 512, 256],
        activation=nn.GELU,
        learning_rate=1e-6,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=25, verbose=True, mode="min")
    logger = TensorBoardLogger(save_dir="FullyINR_logs", version=f"beamforming_{datetime.datetime.now()}")
    # logger = WandbLogger(project="TransformerPerAntennaAndDir_logs", version=f"beamforming_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"FullyINR_logs/{logger.version}", save_top_k=2,
                                          monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    logger.log_graph(model)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # data_path = pathlib.Path("data/new_data")
    data_path = pathlib.Path("new_beamforming_data")
    normal_train()
