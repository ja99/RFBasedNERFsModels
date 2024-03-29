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
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.ReLU(),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            nn.Linear(16384, dim_output),
            nn.ReLU(),
        )

        self.lr = learning_rate
        self.loss_fn = torch.nn.MSELoss()

        # ToDo: save hyperparameters
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
        # Encoding
        grid_values, voltages = batch

        # Encoding
        voltages = torch.flatten(voltages, start_dim=2)

        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        loss = self.loss_fn(output, grid_values[:, :, 2])
        print(loss)

        return output

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        grid_values, voltages = batch

        # Encoding
        voltages = torch.flatten(voltages, start_dim=2)

        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        loss = self.loss_fn(output, grid_values[:, :, 2])

        self.log('train_loss', loss)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        grid_values, voltages = batch

        # Encoding
        voltages = torch.flatten(voltages, start_dim=2)

        antennas_latents = self.encoder(voltages)
        antennas_latents = antennas_latents.flatten(start_dim=1)

        # Decoding
        output = self.decoder(antennas_latents)

        loss = self.loss_fn(output, grid_values[:, :, 2])

        if self.validation_step_counter >= 3:
            self.logger.experiment.add_histogram('actual_values_sdf', grid_values[:, :, 2], self.global_step)
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
        batch_size=64,
        load_positions=False,
        load_normal_grids=False,
        load_subsampled_grids=True,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    data_module.setup()

    batch = next(iter(data_module.test_dataloader()))

    output_size = batch[0].shape[-2]

    model = FullyEndToEnd(
        n_antennas=16,
        n_timesteps=750,
        dim_position=3,
        dim_output=output_size,
        learning_rate=1e-5,
    )

    PATH = "FullySimplifiedNonINR_logs"

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
    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=False,
        load_normal_grids=False,
        load_subsampled_grids=True,
        load_sensor_samples=True,
        split_sensor_samples=True,
        flatten_sensor_samples=False,
        load_occupation_grids=False,
    )

    data_module.setup()

    model = FullyEndToEnd.load_from_checkpoint(
        "FullySimplifiedNonINR_logs/beamforming_2024-03-13 11:58:23.291125/epoch=74-step=525.ckpt"
    )
    model = model.cuda()

    model.eval()

    batch = next(iter(data_module.test_dataloader()))
    batch = [b.cuda() for b in batch]

    output = model(batch, 0)

    original_shape = data_module.dataset.grid_original_shape[:-1]

    output = output[0].view(original_shape)

    ground_truth = batch[0][:, :, 2].view(original_shape)

    visualize_sdf(output)
    visualize_sdf(ground_truth)





if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    data_path = pathlib.Path("new_beamforming_data")
    # normal_train()
    render()
# tensorboard --logdir FullySimplifiedNonINR_logs
