import json
import os
from pathlib import Path
import warnings

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from SaveAsHDF5 import split_scene_indices


class SensorDataset(Dataset):
    """Read one scene at a time. Each DataLoader worker opens its own HDF5 handle."""

    def __init__(self, data_path: Path, load_normal_grids=False, load_subsampled_grids=None,
                 load_occupation_grids=False, load_sensor_samples=True, split_sensor_samples=False,
                 flatten_sensor_samples=True, flatten_grids=True, load_positions=False):
        self.path = Path(data_path) / "dataset.hdf5"
        self.load_positions = load_positions
        self.load_sensor_samples = load_sensor_samples
        self.flatten_grids = flatten_grids
        self.flatten_sensor_samples = flatten_sensor_samples
        self.split_sensor_samples = split_sensor_samples
        self.load_occupation_grids = load_occupation_grids
        self.grid_key = None
        if load_subsampled_grids is None:
            load_subsampled_grids = not (load_normal_grids or load_occupation_grids)
        if load_occupation_grids:
            self.grid_key = "grids"
        elif load_subsampled_grids:
            self.grid_key = "subsampled_grids"
        elif load_normal_grids:
            self.grid_key = "grids"
        if not any((self.grid_key, load_sensor_samples, load_positions)):
            raise ValueError("Select at least one of positions, grids or sensor samples")
        self._file = None
        self._pid = None

        with h5py.File(self.path, "r") as f:
            count_key = self.grid_key or ("sensor_samples" if "sensor_samples" in f else "grids")
            self.count = f[count_key].shape[0]
            self.splits = None
            if "splits" in f:
                self.splits = {key: f["splits"][key][:].tolist() for key in ("train", "validation", "test")}
                all_indices = [i for indices in self.splits.values() for i in indices]
                if sorted(all_indices) != list(range(self.count)):
                    raise ValueError("Stored scene splits must partition the dataset exactly once")
            self.scene_ids = f["scene_ids"].asstr()[:].tolist() if "scene_ids" in f else None
            self.preprocessing_id = f.attrs.get("preprocessing_id")
            self.training_protocol = (json.loads(f["training_protocol"].asstr()[()])
                                      if "training_protocol" in f else None)
            self.waveform_mode = str(f.attrs.get("waveform_mode", "legacy_raw"))
            self.subsample_stride = int(f.attrs.get("subsample_stride", 5))
            self.sdf_zero = None
            self.sdf_scale = None
            if "grid_min" in f.attrs and "grid_scale" in f.attrs:
                self.sdf_scale = float(f.attrs["grid_scale"][2])
                self.sdf_zero = float(np.float32(-f.attrs["grid_min"][2] / self.sdf_scale))
            if self.grid_key:
                shape = f[self.grid_key].shape[1:]
                self.grid_original_shape = (*shape[:-1], 2) if load_occupation_grids else shape
            if load_positions:
                positions = f["positions"][:]
                if self.grid_key == "subsampled_grids":
                    stride = self.subsample_stride
                    full_shape = f["grids"].shape[1:-1]
                    positions = positions.reshape(*full_shape, 3)[::stride, ::stride, ::stride].reshape(-1, 3)
                self.positions = torch.from_numpy(np.ascontiguousarray(positions, dtype=np.float32))
            if "sensor_samples" not in f and not load_sensor_samples:
                return
            sensor_data = f["sensor_samples"]
            if sensor_data.shape[0] != self.count:
                raise ValueError("Scene counts differ for RF recordings and label grids")
            if "sensor_shape" in f.attrs:
                self.sensor_shape = tuple(map(int, f.attrs["sensor_shape"]))
                self.sensor_storage = str(f.attrs["sensor_storage"])
            else:
                # The two historical HDF5 writers produced [N,80,T] and [N,T,A].
                rows, cols = sensor_data.shape[1:]
                if rows == 80:
                    self.sensor_shape = (16, 5, cols)
                    self.sensor_storage = "antenna_direction,time"
                else:
                    self.sensor_shape = (cols, 1, rows)
                    self.sensor_storage = "time,antenna"

    def __len__(self):
        return self.count

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def __getitem__(self, idx):
        if self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        parts = []
        if self.load_positions:
            parts.append(self.positions)
        if self.grid_key:
            values = torch.from_numpy(self._file[self.grid_key][idx]).float()
            if self.load_occupation_grids:
                sdf = values[..., 2]
                # Original archives omitted the SDF transform. Preserve their occupancy convention.
                occupied = values[..., 0] > 0.1 if self.sdf_zero is None else sdf < self.sdf_zero
                values = torch.stack((occupied.to(sdf.dtype), sdf), dim=-1)
            if self.flatten_grids:
                values = values.reshape(-1, values.shape[-1])
            parts.append(values)
        if self.load_sensor_samples:
            signal = torch.from_numpy(self._file["sensor_samples"][idx]).float()
            if self.flatten_sensor_samples:
                signal = signal.flatten()
            elif self.split_sensor_samples:
                if self.sensor_storage == "time,antenna":
                    signal = signal.T.contiguous()
                signal = signal.reshape(self.sensor_shape)
            parts.append(signal)
        return tuple(parts) if len(parts) > 1 else parts[0]


class SensorDataModule(pl.LightningDataModule):
    def __init__(self, data_path: Path, batch_size=1024, load_normal_grids=False, load_subsampled_grids=None,
                 load_occupation_grids=False, load_sensor_samples=True, split_sensor_samples=False,
                 flatten_sensor_samples=True, flatten_grids=True, load_positions=False, *, num_workers=0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = SensorDataset(
            data_path, load_normal_grids=load_normal_grids, load_subsampled_grids=load_subsampled_grids,
            load_occupation_grids=load_occupation_grids, load_sensor_samples=load_sensor_samples,
            split_sensor_samples=split_sensor_samples, flatten_sensor_samples=flatten_sensor_samples,
            flatten_grids=flatten_grids, load_positions=load_positions)

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return
        splits = self.dataset.splits
        if splits is None:
            warnings.warn("Legacy HDF5 has no saved split. Using a reproducible new split because membership in "
                          "historical training runs is unknown.", UserWarning, stacklevel=2)
            splits = split_scene_indices(len(self.dataset))
        self.train_dataset = Subset(self.dataset, splits["train"])
        self.val_dataset = Subset(self.dataset, splits["validation"])
        self.test_dataset = Subset(self.dataset, splits["test"])

    def checkpoint_metric(self):
        self.setup("fit")
        if not len(self.train_dataset):
            raise ValueError("This archive has no training scenes. Use its test split for evaluation.")
        if len(self.val_dataset):
            return "val_loss"
        warnings.warn("No validation scenes are available. Selecting the checkpoint by training loss.",
                      UserWarning, stacklevel=2)
        return "train_loss"

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
