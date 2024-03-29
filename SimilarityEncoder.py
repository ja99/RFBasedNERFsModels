import datetime
import pathlib
import random
import time
from pprint import pprint
from typing import Tuple, Any, Dict, List

from pytorch_lightning.utilities.types import STEP_OUTPUT

from helpers import *

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from siren_pytorch import SirenNet
import plotly.express as px
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F


class SensorDataset(Dataset):
    def __init__(self, data_path: pathlib.Path):
        self.grids = [np.load(path) for path in data_path.glob("*/grid.npy")]
        self.sensor_samples = [np.load(path) for path in data_path.glob("*/sensor_samples.npy")]

        self.grid_size = len(torch.tensor(self.grids[0]).flatten())
        self.sensor_samples_size = len(torch.tensor(self.sensor_samples[0]).flatten())

        # Normalizing the values between 0 and 1 for each of the 3 columns individually
        min_vals = self.grids.min(axis=0)
        max_vals = self.grids.max(axis=0)
        self.grids = (self.grids - min_vals) / (max_vals - min_vals)

        # Normalizing the values between 0 and 1
        min_vals = self.sensor_samples.min(axis=0)
        max_vals = self.sensor_samples.max(axis=0)
        self.sensor_samples = (self.sensor_samples - min_vals) / (max_vals - min_vals)

    def __len__(self):
        return len(self.grids)

    def __getitem__(
            self,
            idx: int
    ) -> dict[str, torch.Tensor]:
        #ToDo: results in data leakage, fix this
        rand_other_idx = random.randint(0, len(self.grids) - 1)

        return {
            "grid_a": torch.tensor(self.grids[idx]).flatten().float(),
            "grid_b": torch.tensor(self.grids[rand_other_idx]).flatten().float(),
            "sensor_samples_a": torch.tensor(self.sensor_samples[idx]).flatten().float(),
            "sensor_samples_b": torch.tensor(self.sensor_samples[rand_other_idx]).flatten().float(),
        }




class SensorDataModule(pl.LightningDataModule):
    def __init__(self, data_path: pathlib.Path, batch_size: int = 1024):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = SensorDataset(self.data_path)

    def prepare_data(self):
        # Load the data once and only once (no state or downloads here)
        ...

    def setup(self, stage=None):
        # Split dataset
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                                             [int(len(self.dataset) * 0.8),
                                                                              len(self.dataset) - int(
                                                                                  len(self.dataset) * 0.8)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


# Define the LightningModule
class EncoderModule(pl.LightningModule):
    def __init__(self,
                 sensor_samples_size: int = 1000,  # n_antenna*n_sample
                 dims_hidden: List[int] = [256],
                 activation: nn.Module = None,
                 learning_rate: float = 1e-6,
                 ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(sensor_samples_size, dims_hidden[0]))
        layers.append(activation())
        for i in range(1, len(dims_hidden)):
            layers.append(nn.Linear(dims_hidden[i-1], dims_hidden[i]))
            layers.append(activation())

        self.encode = nn.Sequential(*layers)


        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self,
                inputs: torch.Tensor,  # [B, n_antennas*n_samples]
                ) -> torch.Tensor:  # [B, dims_hidden[-1]]
        latent = self.encode(inputs)
        return latent


    def training_step(self,
                      inputs: Dict[str, torch.Tensor],
                      batch_idx: int
                      ):
        grid_a, grid_b, sensor_samples_a, sensor_samples_b = inputs["grid_a"], inputs["grid_b"], inputs["sensor_samples_a"], inputs["sensor_samples_b"]

        latent_a = self(sensor_samples_a)
        latent_b = self(sensor_samples_b)

        cosine_similarity_latents = F.cosine_similarity(latent_a, latent_b, dim=1)
        cosine_similarity_grids = F.cosine_similarity(grid_a, grid_b, dim=1)

        loss = self.loss_fn(cosine_similarity_latents, cosine_similarity_grids)
        self.log('train_loss', loss)
        return loss



    def validation_step(self,
                      inputs: Dict[str, torch.Tensor],
                      batch_idx: int
                      ):
        grid_a, grid_b, sensor_samples_a, sensor_samples_b = inputs["grid_a"], inputs["grid_b"], inputs["sensor_samples_a"], inputs["sensor_samples_b"]

        latent_a = self(sensor_samples_a)
        latent_b = self(sensor_samples_b)

        cosine_similarity_latents = F.cosine_similarity(latent_a, latent_b, dim=1)
        cosine_similarity_grids = F.cosine_similarity(grid_a, grid_b, dim=1)

        loss = self.loss_fn(cosine_similarity_latents, cosine_similarity_grids)
        self.log('val_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([*self.encode.parameters()], lr=self.learning_rate)
        return optimizer






def normal_train():
    data_module = SensorDataModule(data_path, batch_size=64)
    model = EncoderModule(
        sensor_samples_size=data_module.dataset.sensor_samples_size,
        dims_hidden=[2048, 512],
        activation=nn.Tanh,
        learning_rate=1e-6
    )

    trainer = pl.Trainer(
        max_epochs=100,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)



if __name__ == "__main__":
    data_path = pathlib.Path("data/new_data")
    normal_train()

# command to start tensorboard: tensorboard --logdir=runs
