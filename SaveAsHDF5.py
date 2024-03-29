import numpy as np
import h5py
from pathlib import Path
from helpers import create_3d_grid_indices
import torch

def save_to_hdf5_beamforming(data_path:Path):
    grids = np.array([np.load(path) for path in data_path.glob("*/grid.npy")], dtype=np.float64)
    sensor_samples = np.array([np.load(path) for path in data_path.glob("*/sensor_samples.npy")],
                                   dtype=np.float64)

    print("loaded data")
    print(grids.shape)
    grids = grids[:,:-15,...] # remove the last 15 samples from x direction

    sensor_samples = sensor_samples.reshape(sensor_samples.shape[0], 16*5, 750)

    # Normalize each of the 3 channels individually
    for i in range(3):  # Iterate over the last dimension
        min_val = grids[..., i].min()
        max_val = grids[..., i].max()
        grids[..., i] = (grids[..., i] - min_val) / (max_val - min_val)
    grids = np.float32(grids)

    print("normalized grids")

    subsampled_grids = grids[:, ::5, ::5, ::5]

    # Normalizing the values between 0 and 1
    min_vals = sensor_samples.min()
    max_vals = sensor_samples.max()
    sensor_samples = (sensor_samples - min_vals) / (max_vals - min_vals)
    sensor_samples = np.float32(sensor_samples)

    print("normalized sensor samples")

    positions = torch.tensor(
        create_3d_grid_indices(grids.shape[1], grids.shape[2], grids.shape[3])).float()
    min_vals = positions.min()
    max_vals = positions.max()
    positions = (positions - min_vals) / (max_vals - min_vals)
    positions = positions.cpu().numpy()

    print("normalized positions")

    with h5py.File(data_path/"dataset.hdf5", 'w') as h5file:
        h5file.create_dataset('grids', data=grids)
        h5file.create_dataset('subsampled_grids', data=subsampled_grids)
        h5file.create_dataset('sensor_samples', data=sensor_samples)
        h5file.create_dataset('positions', data=positions)

def save_to_hdf5(data_path:Path):
    grids = np.array([np.load(path) for path in data_path.glob("*/grid.npy")], dtype=np.float64)
    sensor_samples = np.array([np.load(path) for path in data_path.glob("*/sensor_samples.npy")],
                                   dtype=np.float64)

    # Compute min and max for each of the last three properties across all grids
    min_values = grids.min(axis=(0, 1, 2, 3))
    max_values = grids.max(axis=(0, 1, 2, 3))
    # Normalize each value
    grids = (grids - min_values) / (max_values - min_values)
    grids = np.float32(grids)

    subsampled_grids = grids[:, ::5, ::5, ::5]

    # Normalizing the values between 0 and 1
    min_vals = sensor_samples.min()
    max_vals = sensor_samples.max()
    sensor_samples = (sensor_samples - min_vals) / (max_vals - min_vals)
    # reorder to (b, time_step, sensor)
    sensor_samples = np.transpose(sensor_samples, (0, 2, 1))
    sensor_samples = np.float32(sensor_samples)

    positions = torch.tensor(
        create_3d_grid_indices(grids.shape[1], grids.shape[2], grids.shape[3])).float()
    min_vals = positions.min()
    max_vals = positions.max()
    positions = (positions - min_vals) / (max_vals - min_vals)
    positions = positions.cpu().numpy()

    with h5py.File(data_path/"dataset.hdf5", 'w') as h5file:
        h5file.create_dataset('grids', data=grids)
        h5file.create_dataset('subsampled_grids', data=subsampled_grids)
        h5file.create_dataset('sensor_samples', data=sensor_samples)
        h5file.create_dataset('positions', data=positions)

# # Paths
# data_path = Path('data')
#
# # Save to HDF5
# save_to_hdf5(data_path)

data_path = Path('/home/janis/Downloads/new_beamforming_data')
save_to_hdf5_beamforming(data_path)