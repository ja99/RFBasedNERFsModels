import math
from pathlib import Path
import warnings

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F

from CustomDataset import SensorDataset as Scenes, SensorDataModule as SceneDataModule
from field_training import DEFAULT_DATA_PATH, SensorModel, run_cli


class SensorDataset(Dataset):
    """Choose both members of a similarity pair from the same scene split."""
    def __init__(self, data_path, *, deterministic=False):
        self.scenes = data_path
        if isinstance(data_path, (str, Path)):
            self.scenes = Scenes(data_path, load_normal_grids=True)
        source = self.scenes.dataset if isinstance(self.scenes, Subset) else self.scenes
        self.waveform_mode = source.waveform_mode
        self.preprocessing_id = source.preprocessing_id
        self.training_protocol = source.training_protocol
        self.grid_size = math.prod(source.grid_original_shape)
        self.sensor_samples_size = math.prod(source.sensor_shape)
        self.deterministic = deterministic

    def __len__(self):
        return len(self.scenes) if len(self.scenes) >= 2 else 0

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        offset = 1
        if not self.deterministic:
            offset = torch.randint(1, len(self.scenes), ()).item()
        a, va = self.scenes[idx]
        b, vb = self.scenes[(idx + offset) % len(self.scenes)]
        return {"grid_a": a.flatten(), "grid_b": b.flatten(),
                "sensor_samples_a": va.flatten(), "sensor_samples_b": vb.flatten()}


class SensorDataModule(SceneDataModule):
    def __init__(self, data_path, batch_size=4):
        super().__init__(data_path, batch_size=batch_size, load_normal_grids=True, load_subsampled_grids=False)

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return
        super().setup(stage)
        if len(self.val_dataset) == 1:
            warnings.warn("Similarity validation requires two distinct scenes. This split has no validation pairs.",
                          UserWarning, stacklevel=2)
        self.train_dataset = SensorDataset(self.train_dataset)
        self.val_dataset = SensorDataset(self.val_dataset, deterministic=True)
        self.test_dataset = SensorDataset(self.test_dataset, deterministic=True)

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError("Similarity training requires at least two distinct training scenes")
        return super().train_dataloader()


class EncoderModule(SensorModel):
    def __init__(self, sensor_samples_size: int = 1000, dims_hidden=(256,),
                 activation=nn.Tanh, learning_rate: float = 1e-6):
        super().__init__()
        if isinstance(activation, str):
            activation = getattr(nn, activation)
        self.save_hyperparameters(ignore=["activation"])
        self.hparams.activation = activation.__name__

        layers = []
        width = sensor_samples_size
        for next_width in dims_hidden:
            layers.extend([nn.Linear(width, next_width), activation()])
            width = next_width
        self.encode = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, inputs):
        return self.encode(inputs)

    def _loss(self, batch):
        latent_a = self(batch["sensor_samples_a"])
        latent_b = self(batch["sensor_samples_b"])
        return self.loss_fn(F.cosine_similarity(latent_a, latent_b, dim=1),
                            F.cosine_similarity(batch["grid_a"], batch["grid_b"], dim=1))

    def training_step(self, inputs, batch_idx):
        loss = self._loss(inputs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, inputs, batch_idx):
        loss = self._loss(inputs)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.encode.parameters(), lr=self.learning_rate)


def normal_train(data_path=DEFAULT_DATA_PATH):
    data_module = SensorDataModule(data_path)
    model = EncoderModule(
        sensor_samples_size=math.prod(data_module.dataset.sensor_shape),
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
    run_cli(normal_train)
