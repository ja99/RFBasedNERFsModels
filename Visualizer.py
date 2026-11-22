from typing import Optional

import numpy as np
import pyvista as pv
import torch


def subsample(data: torch.Tensor, subsample_rate: int) -> torch.Tensor:
    return torch.nn.AvgPool3d(subsample_rate, subsample_rate)(data.unsqueeze(0))[0]


def sdf_metrics(prediction, target, sdf_zero=None, sdf_scale=None):
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction {prediction.shape} and target {target.shape} must align")
    delta = (prediction - target).abs()
    scores = {"normalized_mae": delta.mean().item()}
    if sdf_zero is not None:
        predicted = prediction < sdf_zero
        occupied = target < sdf_zero
        intersection = (predicted & occupied).sum().item()
        pred_count, target_count = predicted.sum().item(), occupied.sum().item()
        union = pred_count + target_count - intersection
        scores.update(occupancy_iou=intersection / union if union else 1.0, occupied_predicted=pred_count,
                      occupied_target=target_count, occupied_intersection=intersection, occupied_union=union)
        if sdf_scale is not None:
            scores["sdf_mae_cells"] = scores["normalized_mae"] * sdf_scale
    return scores


def visualize_sdf(data: torch.Tensor, label: str = "SDF", *, sdf_zero=None):
    values = data.detach().cpu().numpy()
    grid = pv.wrap(values)
    plotter = pv.Plotter()
    plotter.add_mesh(grid.outline(), color="gray")
    if sdf_zero is None:
        # Legacy archives omit the physical zero. Use one absolute normalized scale for both fields.
        plotter.add_mesh(grid.slice_orthogonal(), clim=(0, 1), scalar_bar_args={"title": "Normalized SDF"})
        label += " (physical zero unavailable)"
    elif np.min(values) >= sdf_zero:
        label += " (no occupied voxels)"
    elif np.max(values) < sdf_zero:
        plotter.add_mesh(grid.extract_surface(), color="teal")
        label += " (all voxels occupied)"
    else:
        surface = grid.contour([sdf_zero])
        if surface.n_points:
            plotter.add_mesh(surface, color="teal", show_scalar_bar=False)
    plotter.add_text(label)
    plotter.show_bounds()
    plotter.show()


def visualize_sdf_pair(prediction, target, dataset):
    scores = sdf_metrics(prediction, target, dataset.sdf_zero, dataset.sdf_scale)
    print("Whole-scene SDF:", ", ".join(f"{name}={value:.5g}" for name, value in scores.items()))
    visualize_sdf(prediction, "Predicted SDF", sdf_zero=dataset.sdf_zero)
    visualize_sdf(target, "Ground truth SDF", sdf_zero=dataset.sdf_zero)


def visualize_permittivity(data: torch.Tensor, label: str = "Permittivity"):
    grid = pv.wrap(data.detach().cpu().numpy())
    plotter = pv.Plotter()
    plotter.add_volume(grid, clim=(0, 1), opacity=np.linspace(0., 1., 256),
                       scalar_bar_args={"title": "Normalized permittivity"})
    plotter.add_text(label)
    plotter.show_bounds()
    plotter.show()


def visualize_objs(export_grid: np.ndarray, subsampling: Optional[int]):
    values = torch.as_tensor(export_grid[..., 0])
    if subsampling:
        values = subsample(values, subsampling)
    pv.wrap(values.cpu().numpy()).plot(volume=True, show_bounds=True)
