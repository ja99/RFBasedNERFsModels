import datetime
import math
import pathlib
from typing import Tuple

from Visualizer import visualize_sdf
from helpers import sample_indices
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import CustomDataset


class FullyEndToEnd(pl.LightningModule):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16):
        super(FullyEndToEnd, self).__init__()

        # encoding per t?
        # encoding per antenna?
        # encoding per t and antenna?
        # encoding per direction?
        # adding latents up?

        # in size: B x 16 x 5 x 750
        # in size: B x n_antennas x dir x timestep

        # add each dir individually
        self.encoder = nn.Sequential(
            nn.Linear(5 * n_timesteps, 8192),
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

        # concatenate with position
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
        self.loss_fn = torch.nn.MSELoss()

        # ToDo: save hyperparameters
        self.save_hyperparameters(
            "n_antennas",
            "n_timesteps",
            "dim_position",
            "dim_output",
            "learning_rate",
            "internal_batch_size_training",
            "internal_batch_size_validation",
        )

        self.validation_step_counter = 0
        # self.logger.log_graph(self, (torch.zeros((1, n_timesteps, n_antennas)), torch.zeros((1, dim_position))))

    def forward(self,
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        positions, grid_values, voltages = batch

        positions = positions.squeeze()
        grid_values = grid_values.squeeze()
        voltages = voltages.squeeze()

        # Encoding
        voltages = torch.flatten(voltages, start_dim=1).unsqueeze(0)
        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)
        antennas_latents = antennas_latents.expand(positions.shape[0], -1).squeeze()


        # Decoding
        decoder_input = torch.cat((positions, antennas_latents), dim=-1)

        output = self.decoder(decoder_input)

        loss = self.loss_fn(output, grid_values[:, 1].unsqueeze(1))

        return output

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        positions, grid_values, voltages = batch

        # print(f"{positions.shape=}")
        # print(f"{grid_values.shape=}")
        # print(f"{voltages.shape=}")

        positions = positions.squeeze()
        grid_values = grid_values.squeeze()
        voltages = voltages.squeeze()

        # subsample randomly
        indicies = sample_indices(grid_values, self.internal_batch_size_training, by_index=1)
        positions = positions[indicies, :]
        grid_values = grid_values[indicies, :]

        # Encoding
        voltages = torch.flatten(voltages, start_dim=1).unsqueeze(0)
        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)
        antennas_latents = antennas_latents.expand(self.internal_batch_size_training, -1).squeeze()

        # Decoding
        decoder_input = torch.cat((positions, antennas_latents), dim=-1)

        output = self.decoder(decoder_input)

        loss = self.loss_fn(output, grid_values[:, 1].unsqueeze(1))

        self.log('train_loss', loss)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        positions, grid_values, voltages = batch

        # print(f"{positions.shape=}")
        # print(f"{grid_values.shape=}")
        # print(f"{voltages.shape=}")

        positions = positions.squeeze()
        grid_values = grid_values.squeeze()
        voltages = voltages.squeeze()

        # subsample randomly
        indicies = sample_indices(grid_values, self.internal_batch_size_training, by_index=1)
        positions = positions[indicies, :]
        grid_values = grid_values[indicies, :]

        # Encoding
        voltages = torch.flatten(voltages, start_dim=1).unsqueeze(0)
        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)
        antennas_latents = antennas_latents.expand(self.internal_batch_size_training, -1).squeeze()

        # Decoding
        decoder_input = torch.cat((positions, antennas_latents), dim=-1)

        output = self.decoder(decoder_input)

        loss = self.loss_fn(output, grid_values[:, 1].unsqueeze(1))

        if self.validation_step_counter >= 3:
            # draw histogram of predicted values and grid values for evaluation
            # self.logger.experiment.add_histogram('actual_values_occupancy', grid_values[:, 0], self.global_step)
            # self.logger.experiment.add_histogram('predicted_values_occupancy', output[:, 0], self.global_step)
            self.logger.experiment.add_histogram('actual_values_sdf', grid_values[:, 1], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_sdf', output, self.global_step)

        self.validation_step_counter += 1
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Choose an optimizer and, optionally, learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def normal_train():
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=False,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=True,
    )

    model = FullyEndToEnd(
        n_antennas=16,
        n_timesteps=750,
        dim_position=3,
        dim_output=1,
        learning_rate=1e-6,
        internal_batch_size_training=32,
        internal_batch_size_validation=32,
    )

    PATH = "FullySimplifiedEncodedPerAntenna_logs"

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=25, verbose=True, mode="min")
    logger = TensorBoardLogger(save_dir=PATH, version=f"beamforming_{datetime.datetime.now()}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"{PATH}/{logger.version}", save_top_k=1,
                                          monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=1000,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    logger.log_graph(model)

    trainer.fit(model, datamodule=data_module)


def render():
    N_PER_RENDER = 16384

    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=False,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=True,
    )

    data_module.setup()

    model = FullyEndToEnd.load_from_checkpoint(
        "FullySimplifiedEncodedPerAntenna_logs/beamforming_2024-03-13 14:53:30.293106/epoch=104-step=42525.ckpt"
    )
    model = model.cuda()

    model.eval()
    model.requires_grad_(False)

    batch = next(iter(data_module.test_dataloader()))
    batch = [b.cuda() for b in batch]

    positions, grid_values, voltages = batch

    output = torch.zeros((positions.shape[0], positions.shape[1], 1)).cuda()

    i = 0
    while i < positions.shape[1]:
        j = i + N_PER_RENDER
        if j > positions.shape[1]:
            j = positions.shape[1]
        output[:, i:j, :] = model((positions[:, i:j, :], grid_values[:, i:j, :], voltages), 0)
        i = j
        model.zero_grad()
        print(f"{i/positions.shape[1]:.2f}")


    original_shape = data_module.dataset.grid_original_shape[:-1]

    output = output[0].view(original_shape)

    ground_truth = batch[1][:, :, 1].view(original_shape)

    visualize_sdf(output)
    visualize_sdf(ground_truth)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    data_path = pathlib.Path("new_beamforming_data")
    # normal_train()
    render()
# tensorboard --logdir FullySimplifiedEncodedPerAntenna_logs
