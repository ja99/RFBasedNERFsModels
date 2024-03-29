import pathlib
from typing import Tuple, Union
import pytorch_lightning as pl
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SensorDataset(Dataset):
    def __init__(self,
                 data_path: pathlib.Path,
                 load_normal_grids: bool = False,
                 load_subsampled_grids: bool = True,
                 load_occupation_grids: bool = False,
                 load_sensor_samples: bool = True,
                 split_sensor_samples: bool = False,
                 flatten_sensor_samples: bool = True,
                 flatten_grids: bool = True,
                 load_positions: bool = False,
                 ):
        with h5py.File(data_path / "dataset.hdf5", 'r') as h5file:
            if load_normal_grids:
                self.grids = np.array(h5file['grids'])
                print("loaded normal grids")
            if load_subsampled_grids:
                self.grids = np.array(h5file['subsampled_grids'])
                print("loaded subsampled grids")
            if load_occupation_grids:
                grids = np.array(h5file['grids'])
                occupated = (grids[..., 0] > 0.1).astype(float)
                sdf = grids[..., 2]
                self.grids = np.stack([occupated, sdf], axis=-1)

                print("loaded occupation grids")
            if load_sensor_samples:
                self.sensor_samples = np.array(h5file['sensor_samples'])
                print("loaded sensor samples")
            if load_positions:
                self.positions = np.array(h5file['positions'])
                self.positions = torch.tensor(self.positions).float()
                print("loaded positions")

        if load_normal_grids or load_subsampled_grids or load_occupation_grids:
            self.grid_original_shape = self.grids[0].shape
            if flatten_grids:
                self.grids = torch.tensor(self.grids).flatten(start_dim=1, end_dim=-2).float()
            else:
                self.grids = torch.tensor(self.grids).float()
        if load_sensor_samples:
            if flatten_sensor_samples:
                self.sensor_samples = torch.tensor(self.sensor_samples).flatten(start_dim=1)
            elif split_sensor_samples:
                sensor_samples = torch.tensor(self.sensor_samples)
                B, n_mul_5, _ = sensor_samples.shape  # This gives you B and n*5
                n = n_mul_5 // 5  # Assuming n*5 is perfectly divisible by 5

                # Now reshape the tensor
                self.sensor_samples = sensor_samples.view(B, n, 5, 750)
            else:
                self.sensor_samples = torch.tensor(self.sensor_samples)

        self.load_positions = load_positions
        self.load_normal_grids = load_normal_grids
        self.load_subsampled_grids = load_subsampled_grids
        self.load_sensor_samples = load_sensor_samples

    def __len__(self):
        if self.load_subsampled_grids or self.load_normal_grids:
            return self.grids.shape[0]
        elif self.load_sensor_samples:
            return self.sensor_samples.shape[0]
        else:
            raise ValueError("No data loaded")

    def __getitem__(
            self,
            idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:  # (grid, sensor_samples):
        if self.load_positions:
            return self.positions, self.grids[idx], self.sensor_samples[idx]
        elif self.load_subsampled_grids or self.load_normal_grids:
            return self.grids[idx], self.sensor_samples[idx]
        else:
            return self.sensor_samples[idx]


class SensorDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: pathlib.Path,
                 batch_size: int = 1024,
                 load_normal_grids: bool = False,
                 load_subsampled_grids: bool = True,
                 load_occupation_grids: bool = False,
                 load_sensor_samples: bool = True,
                 split_sensor_samples: bool = False,
                 flatten_sensor_samples: bool = True,
                 flatten_grids: bool = True,
                 load_positions: bool = False,
                 ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.dataset = SensorDataset(
            self.data_path,
            load_normal_grids=load_normal_grids,
            load_subsampled_grids=load_subsampled_grids,
            load_occupation_grids=load_occupation_grids,
            load_sensor_samples=load_sensor_samples,
            split_sensor_samples=split_sensor_samples,
            flatten_sensor_samples=flatten_sensor_samples,
            flatten_grids=flatten_grids,
            load_positions=load_positions,
        )

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
