from typing import Optional
import pyvista as pv
import torch
import plotly.express as px
import numpy as np
import h5py


def subsample(data: torch.Tensor, subsample_rate: int) -> torch.Tensor:
    data = torch.unsqueeze(data, 0)
    data = torch.nn.AvgPool3d(subsample_rate, subsample_rate)(data)[0]
    return data


# ToDo: make real time possible
# ToDo: add a slider to step through the time steps
def visualize_sdf(data: torch.Tensor, label: str = "SDF"):
    opacity = [1, 0, 0, 0, 0]

    data = data.detach().cpu().numpy()
    print(f"Data shape: {data.shape}")

    data = pv.wrap(data)

    # data.plot(volume=True, opacity=opacity)  # Volume render
    # plotter=pv.Plotter()
    # plotter.add_volume(data, opacity='sigmoid_10')
    # plotter.show(auto_close=False)
    data.plot(volume=True, show_bounds=True, opacity=opacity, text=label, interactive=True)  # Volume render
    print("visualized sdf")


def visualize_permittivity(data: torch.Tensor):
    opacity = [0, 0, 0, 0, 1]

    data = data.detach().cpu().numpy()
    print(f"Data shape: {data.shape}")

    data = pv.wrap(data)

    # data.plot(volume=True, opacity=opacity)  # Volume render
    # plotter=pv.Plotter()
    # plotter.add_volume(data, opacity='sigmoid_10')
    # plotter.show(auto_close=False)
    data.plot(volume=True, show_bounds=True, opacity=opacity)  # Volume render
    print("visualized permittivity")


def visualize_objs(
        export_grid: np.ndarray,  # [x, y, z, permittivity, conductivity]
        subsampling: Optional[int]
):
    # opacity = [0, 0.01, 0.2, 0.7, 1.0]

    export_grid = export_grid[:, :, :, 0]

    if subsampling:
        export_grid = subsample(torch.tensor(export_grid), subsampling).cpu().numpy()

    export_grid = pv.wrap(export_grid)
    # export_grid.plot(volume=True, opacity=opacity)  # Volume render
    export_grid.plot(volume=True, show_bounds=True)  # Volume render


if __name__ == "__main__":
    grids = h5py.File("new_beamforming_data/dataset.hdf5", 'r')['grids']
    grid = torch.tensor(grids[5])
    visualize_permittivity(grid[:, :, :, 0])
    visualize_sdf(grid[:, :, :, 2])
