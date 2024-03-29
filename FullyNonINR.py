import datetime
import math
import pathlib
from typing import Tuple
from icecream import ic
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
            nn.Linear(5 * n_timesteps + 3, 8192),
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
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, dim_output),
            nn.ReLU(),
        )

        self.lr = learning_rate
        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters(
            "n_antennas",
            "n_timesteps",
            "dim_position",
            "dim_output",
            "learning_rate",
        )

        self.validation_step_counter = 0
        # self.logger.log_graph(self, (torch.zeros((1, n_timesteps, n_antennas)), torch.zeros((1, dim_position))))

    def forward(self,
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        positions, _, voltages = batch

        positions = positions.permute(1, 0, 2)

        voltages = torch.flatten(voltages, start_dim=2)
        positions = positions.expand(-1, voltages.shape[1], -1)
        voltages = voltages.expand(positions.shape[0], -1, -1)

        # Encoding
        network_input = torch.cat((voltages, positions), dim=2)
        antennas_latents = self.encoder(network_input)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        return output

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        positions, grid_values, voltages = batch

        # subsample randomly
        indices = torch.randint(0, positions.shape[0] - 1, size=(positions.shape[0], 1), device=positions.device)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)

        positions = torch.gather(positions, 1, indices_expanded)
        grid_values = torch.gather(grid_values, 1, indices_expanded)

        positions = positions.flatten(start_dim=1)
        grid_values = grid_values.flatten(start_dim=1, end_dim=-1)

        voltages = torch.flatten(voltages, start_dim=2)
        positions = positions.unsqueeze(1)
        positions = positions.expand(-1, voltages.shape[1], -1)

        # Encoding
        network_input = torch.cat((voltages, positions), dim=2)
        antennas_latents = self.encoder(network_input)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        loss = self.loss_fn(output, grid_values)

        self.log('train_loss', loss)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        positions, grid_values, voltages = batch

        # subsample randomly
        indices = torch.randint(0, positions.shape[0] - 1, size=(positions.shape[0], 1), device=positions.device)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 3)

        positions = torch.gather(positions, 1, indices_expanded)
        grid_values = torch.gather(grid_values, 1, indices_expanded)

        positions = positions.flatten(start_dim=1)
        grid_values = grid_values.flatten(start_dim=1, end_dim=-1)

        voltages = torch.flatten(voltages, start_dim=2)
        positions = positions.unsqueeze(1)
        positions = positions.expand(-1, voltages.shape[1], -1)

        # Encoding
        network_input = torch.cat((voltages, positions), dim=2)
        antennas_latents = self.encoder(network_input)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        loss = self.loss_fn(output, grid_values)

        if self.validation_step_counter >= 3:
            self.logger.experiment.add_histogram('actual_values_permittivity', grid_values[:, 0], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_permittivity', output[:, 0], self.global_step)
            self.logger.experiment.add_histogram('actual_values_conductivity', grid_values[:, 1], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_conductivity', output[:, 1], self.global_step)
            self.logger.experiment.add_histogram('actual_values_sdf', grid_values[:, 2], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_sdf', output[:, 2], self.global_step)

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
        batch_size=256,
        load_positions=True,
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    model = FullyEndToEnd(
        n_antennas=16,
        n_timesteps=750,
        dim_position=3,
        dim_output=3,
        learning_rate=1e-6,
    )

    PATH = "FullyNonINR_logs"

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
        load_normal_grids=True,
        load_subsampled_grids=False,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )
    data_module.setup()

    model = FullyEndToEnd.load_from_checkpoint(
        "FullyNonINR_logs/beamforming_2024-03-13 17:30:54.145985/epoch=59-step=120.ckpt"
    )
    model = model.cuda()

    model.eval()
    model.requires_grad_(False)

    batch = next(iter(data_module.test_dataloader()))
    batch = [b.cuda() for b in batch]

    positions, _, voltages = batch

    print(f"positions: {positions.shape}")

    output = torch.zeros((positions.shape[1], 3)).cuda()

    i = 0
    while i < positions.shape[1]:
        j = i + N_PER_RENDER
        if j > positions.shape[1]:
            j = positions.shape[1]
        out = model((positions[:, i:j, :], "unneeded grid tensor", voltages), 0)
        output[i:j, :] = out
        i = j
        model.zero_grad()
        print(f"{i / positions.shape[1]:.2f}")

    original_shape = data_module.dataset.grid_original_shape

    output = output.view(original_shape)

    ground_truth = batch[1].view(original_shape)

    visualize_sdf(output[:, :, :, 2])
    visualize_sdf(ground_truth[:, :, :, 2])


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    data_path = pathlib.Path("new_beamforming_data")
    # normal_train()
    render()
# tensorboard --logdir FullyNonINR_logs
