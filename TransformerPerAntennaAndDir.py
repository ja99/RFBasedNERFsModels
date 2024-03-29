import datetime
import math
import pathlib
import random
from enum import Enum
from typing import Tuple
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from Visualizer import visualize_sdf, visualize_permittivity
from helpers import sample_indices

import CustomDataset


class PositionalEncodingType(Enum):
    sinusoidal = 1
    learned = 2


class TransformerEncoder(pl.LightningModule):
    def __init__(self,
                 n_antennas: int,
                 n_timesteps: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 position_encoding: PositionalEncodingType = PositionalEncodingType.learned
                 ):
        super().__init__()

        self.encoder_embedding = nn.Linear(n_timesteps, d_model)

        if position_encoding == PositionalEncodingType.sinusoidal:
            self.positional_encoding = self.create_sinusoidal_positional_encoding(n_antennas, d_model)
        elif position_encoding == PositionalEncodingType.learned:
            self.positional_encoding = nn.Embedding(n_antennas, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.position_encoding = position_encoding

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        B, a, t = src.shape
        src_reshaped = src.view(-1, t)

        y = self.encoder_embedding(src_reshaped)
        src = y.view(B, a, -1)

        # Add positional encoding
        if self.position_encoding == PositionalEncodingType.sinusoidal:
            src = src + self.positional_encoding[:a, :]
        elif self.position_encoding == PositionalEncodingType.learned:
            src = src + self.positional_encoding(torch.arange(a, device=src.device))

        memory = self.transformer_encoder(src)

        return memory

    def create_sinusoidal_positional_encoding(self, n_timesteps: int, d_model: int) -> torch.Tensor:
        position = torch.arange(n_timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        sinusoidal_encoding = torch.zeros(n_timesteps, d_model)
        sinusoidal_encoding[:, 0::2] = torch.sin(position * div_term)
        sinusoidal_encoding[:, 1::2] = torch.cos(position * div_term)

        return sinusoidal_encoding.to("cuda")


class CustomTransformer(pl.LightningModule):
    def __init__(self, n_antennas: int, n_timesteps: int, dim_position: int, dim_output: int, d_model: int, nhead: int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1,
                 learning_rate: float = 1e-3, internal_batch_size_training: int = 16,
                 internal_batch_size_validation: int = 16,
                 position_encoding: PositionalEncodingType = PositionalEncodingType.learned
                 ):
        """
        Custom Transformer model built using PyTorch Lightning.

        Args:
            n_antennas (int): Number of antennas in the input data.
            n_timesteps (int): Number of time steps in the input data.
            dim_position (int): Dimension of the position/query tensor.
            dim_output (int): Dimension of the output tensor.
            d_model (int): The dimension of the model (commonly used in transformer models).
            nhead (int): Number of heads in the multiheadattention models.
            num_encoder_layers (int): Number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): Number of sub-decoder-layers in the decoder.
            dim_feedforward (int): Dimension of the feedforward network model.
            dropout (float): Dropout value.
            learning_rate (float): Learning rate for the optimizer.
            internal_batch_size (int): Batch size to use internally in the training step.
        """
        super(CustomTransformer, self).__init__()

        self.encoder = TransformerEncoder(n_antennas=n_antennas,
                                          n_timesteps=n_timesteps,
                                          d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          position_encoding=position_encoding
                                          )

        # Decoder part
        self.decoder_embedding = nn.Linear(dim_position, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(d_model, dim_output)

        self.lr = learning_rate
        self.internal_batch_size_training = internal_batch_size_training
        self.internal_batch_size_validation = internal_batch_size_validation
        self.loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters(
            "n_antennas",
            "n_timesteps",
            "dim_position",
            "dim_output",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "learning_rate",
            "internal_batch_size_training",
            "internal_batch_size_validation",
        )

        self.validation_step_counter = 0
        self.position_encoding = position_encoding
        # self.logger.log_graph(self, (torch.zeros((1, n_timesteps, n_antennas)), torch.zeros((1, dim_position))))

    def forward(self,
                batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        """
        Forward pass of the transformer.

        Args:
            voltages (torch.Tensor): The input tensor of shape (n_timesteps, n_antennas).
            positions (torch.Tensor): The query tensor of shape (dim_position).

        Returns:
            torch.Tensor: Output tensor of shape (dim_output).
        """

        positions, grid_values, voltages = batch
        positions = positions.squeeze()
        grid_values = grid_values.squeeze()

        # Encoding
        positions_latents = self.decoder_embedding(positions).unsqueeze(1)
        antennas_latents = self.encoder(voltages).expand(positions.shape[0], -1, -1)

        output = self.transformer_decoder(positions_latents, antennas_latents)
        decoded = self.output_layer(output).squeeze()

        loss = self.loss_fn(decoded.unsqueeze(1), grid_values.unsqueeze(1))

        return decoded

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int
                      ) -> torch.Tensor:
        """
        Training step of the LightningModule.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the positions, the grid and the
                sensor samples.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """

        positions, grid_values, voltages = batch

        # subsample randomly
        indicies = sample_indices(grid_values[0], self.internal_batch_size_training)
        positions = positions[:, indicies, :]
        grid_values = grid_values[:, indicies, :]

        # Encoding
        positions_latents = self.decoder_embedding(positions).permute(1, 0, 2)

        antennas_latents = self.encoder(voltages).expand(self.internal_batch_size_training, -1, -1)

        # print(f"{positions_latents.shape=}")
        # print(f"{antennas_latents.shape=}")

        output = self.transformer_decoder(positions_latents, antennas_latents)
        decoded = self.output_layer(output)

        loss = self.loss_fn(decoded, grid_values.permute(1, 0, 2))

        self.log('train_loss', loss)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int
                        ) -> torch.Tensor:
        """
        Training step of the LightningModule.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing the positions, the grid and the
                sensor samples.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """

        positions, grid_values, voltages = batch

        # subsample randomly
        indicies = sample_indices(grid_values[0], self.internal_batch_size_training)
        positions = positions[:, indicies, :]
        grid_values = grid_values[:, indicies, :]

        # Encoding
        positions_latents = self.decoder_embedding(positions).permute(1, 0, 2)

        antennas_latents = self.encoder(voltages).expand(self.internal_batch_size_training, -1, -1)

        output = self.transformer_decoder(positions_latents, antennas_latents)
        decoded = self.output_layer(output)

        loss = self.loss_fn(decoded, grid_values.permute(1, 0, 2))

        if self.validation_step_counter >= 2:
            # draw histogram of predicted values and grid values for evaluation
            self.logger.experiment.add_histogram('actual_values_permittivity', grid_values[:, :, 0], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_permittivity', decoded[:, :, 0], self.global_step)
            self.logger.experiment.add_histogram('actual_values_conductivity', grid_values[:, :, 1], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_conductivity', decoded[:, :, 1], self.global_step)
            self.logger.experiment.add_histogram('actual_values_sdf', grid_values[:, :, 2], self.global_step)
            self.logger.experiment.add_histogram('predicted_values_sdf', decoded[:, :, 2], self.global_step)

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
        load_normal_grids=True,
        load_sensor_samples=True,
        load_subsampled_grids=False,
        flatten_sensor_samples=False,
    )

    model = CustomTransformer(n_antennas=16 * 5,
                              n_timesteps=750,
                              dim_position=3,
                              dim_output=3,
                              d_model=1024,
                              nhead=8,
                              num_encoder_layers=2,
                              num_decoder_layers=4,
                              dim_feedforward=2048,
                              dropout=0.1,
                              learning_rate=1e-6,
                              internal_batch_size_training=64,
                              internal_batch_size_validation=64,
                              position_encoding=PositionalEncodingType.sinusoidal
                              )

    sub_dir = "Sinusoidal" if model.position_encoding == PositionalEncodingType.sinusoidal else "Learned"

    PATH = "TransformerPerAntennaAndDir_logs" + f"/{sub_dir}"

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
    N_PER_RENDER = 10000

    data_module = CustomDataset.SensorDataModule(
        data_path,
        batch_size=1,
        load_positions=True,
        load_normal_grids=True,
        load_sensor_samples=True,
        load_subsampled_grids=False,
        flatten_sensor_samples=False
    )

    data_module.setup()

    model = CustomTransformer.load_from_checkpoint(
        "TransformerPerAntennaAndDir_logs/Learned/beamforming_2024-03-11 14:08:03.883772/epoch=66-step=27135.ckpt"
    )
    model = model.cuda()

    model.eval()
    model.requires_grad_(False)

    batch = next(iter(data_module.test_dataloader()))
    batch = [b.cuda() for b in batch]

    positions, grid_values, voltages = batch

    output = torch.zeros((positions.shape[0], positions.shape[1], 3)).cuda()

    i = 0
    while i < positions.shape[1]:
        j = i + N_PER_RENDER
        if j > positions.shape[1]:
            j = positions.shape[1]
        output[:, i:j, :] = model((positions[:, i:j, :], grid_values[:, i:j, :], voltages), 0)
        i = j
        model.zero_grad()
        print(f"{i / positions.shape[1]:.2f}")

    original_shape = data_module.dataset.grid_original_shape

    output = output[0].view(original_shape)

    ground_truth = batch[1].view(original_shape)

    visualize_sdf(output[:, :, :, 2])
    visualize_sdf(ground_truth[:, :, :, 2])
    visualize_permittivity(output[:, :, :, 0])
    visualize_permittivity(ground_truth[:, :, :, 0])


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    data_path = pathlib.Path("new_beamforming_data")
    # normal_train()
    render()

# tensorboard --logdir TransformerPerAntennaAndDir_logs
