"""Train and evaluate calibrated RF-to-SDF decoders on single- and multi-object dielectric scenes."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from CustomDataset import SensorDataset
from SaveAsHDF5 import _load_trace
from echo_components import component_counts, summarize_components
from field_training import check_preprocessing


def open_archive(path, *, labels):
    path = Path(path)
    if path.name == "dataset.hdf5":
        path = path.parent
    dataset = SensorDataset(path, load_normal_grids=labels, load_subsampled_grids=False,
                            load_sensor_samples=True, split_sensor_samples=True,
                            flatten_sensor_samples=False, flatten_grids=False)
    if dataset.waveform_mode != "calibrated" or not dataset.preprocessing_id or dataset.splits is None:
        raise ValueError("This model requires a calibrated archive with a recorded training protocol and scene split")
    return dataset


def trace_features(waves):
    """Keep relative channel strengths, absolute signal level, pulse shape, and timing on every receiver/shot."""
    peak = waves.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-30)
    relative = waves / peak
    rms = relative.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    unit = relative / rms
    flat = unit.reshape(-1, 1, unit.size(-1))
    phase = F.adaptive_avg_pool1d(flat, 48).reshape(*unit.shape[:3], 48)
    energy = F.adaptive_avg_pool1d(flat.square(), 16).reshape(*unit.shape[:3], 16)
    # The logarithm also retains weak leading edges before the main return.
    return torch.cat((phase, torch.log1p(energy), energy.clamp_min(1e-7).log(),
                      rms.log(), peak.log().expand_as(rms)), dim=-1)


def coordinates(shape, device):
    axes = [torch.arange(n, device=device) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).float()


def object_moments(sdf, xyz):
    """Centroid, equivalent-volume radius and determinant-one Cholesky shape, measured from training labels."""
    mask = (sdf.flatten(1) < 0).double()
    xyz = xyz.double()
    count = mask.sum(1)
    if (count == 0).any():
        raise ValueError("The single-object training model requires occupied voxels in every training scene")
    center = mask @ xyz / count[:, None]
    second = (mask @ (xyz[:, :, None] * xyz[:, None, :]).flatten(1)).reshape(-1, 3, 3) / count[:, None, None]
    # Voxels are solid unit cubes whose within-cell variance is 1/12 on each axis.
    covariance = second - center[:, :, None] * center[:, None, :] + torch.eye(3, device=xyz.device) / 12
    lower = torch.linalg.cholesky(covariance)
    log_diag = lower.diagonal(dim1=1, dim2=2).log()
    lower = lower / log_diag.mean(1).exp()[:, None, None]
    log_diag = lower.diagonal(dim1=1, dim2=2).log()
    radius = (3 * count / (4 * math.pi)).pow(1 / 3)
    return torch.cat((center, radius.log()[:, None], log_diag[:, :2], lower[:, 1, 0:1],
                      lower[:, 2, 0:2]), dim=1).float()


def affine_shape(params):
    a, b = params[:, 4:6].clamp(-1.4, 1.4).unbind(1)
    l10, l20, l21 = params[:, 6:9].clamp(-3., 3.).unbind(1)
    zero = torch.zeros_like(a)
    return torch.stack((a.exp(), zero, zero, l10, b.exp(), zero, l20, l21, (-a-b).exp()), dim=1).reshape(-1, 3, 3)


def base_distance(delta, local, radius):
    distance = torch.linalg.vector_norm(delta, dim=-1)
    metric = torch.linalg.vector_norm(local, dim=-1)
    radial_scale = torch.where(distance > 1e-5, distance / metric.clamp_min(1e-5), 1.)
    return distance - radius[:, None] * radial_scale


def local_training_targets(sdf, moments, size):
    """Sample target surface corrections on a centered, shape-normalized lattice using training SDFs only."""
    shape = sdf.shape[1:]
    axis = torch.linspace(-1.65, 1.65, size, device=sdf.device)
    coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).flatten(0, 2)
    radius = moments[:, 3].exp()
    local = coords[None] * radius[:, None, None]
    delta = local @ affine_shape(moments).transpose(1, 2)
    world = moments[:, None, :3] + delta
    sample_coords = 2 * world / world.new_tensor(shape).sub(1).clamp_min(1) - 1
    sample = sample_coords.flip(-1)[:, :, None, None]
    truth = F.grid_sample(sdf[:, None], sample, align_corners=True, padding_mode="border")[:, 0, :, 0, 0]
    residual = ((truth - base_distance(delta, local, radius)) / radius[:, None]).clamp(-.55, .55)
    weights = (sample_coords.abs() <= 1).all(-1) * (1 / (1 + 4*(coords.norm(dim=-1)-1).abs()))
    return residual, weights


class ApertureEncoder(nn.Module):
    """Encode temporal RF returns while preserving receiver and shot order."""

    def __init__(self, feature_shape, grid_shape, hidden):
        super().__init__()
        antennas, shots, features = feature_shape
        self.config = dict(feature_shape=list(feature_shape), grid_shape=list(grid_shape), hidden=hidden)
        self.register_buffer("feature_mean", torch.zeros(feature_shape))
        self.register_buffer("feature_scale", torch.ones(feature_shape))
        self.trace_encoder = nn.Sequential(nn.Linear(features, 32), nn.LayerNorm(32), nn.SiLU(), nn.Linear(32, 24))
        self.shot_encoder = nn.Sequential(nn.Linear(shots * 24, 48), nn.LayerNorm(48), nn.SiLU())
        self.aperture_encoder = nn.Sequential(nn.Linear(antennas * 48, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                              nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())

    def standardize(self, features):
        return ((features - self.feature_mean) / self.feature_scale).clamp(-8, 8)

    def fit_readout(self, features, target, ridge):
        self.feature_mean.copy_(features.mean(0))
        spread = features.std(0, unbiased=False)
        floor = spread.flatten(0, 1).median(0).values.clamp_min(1e-3) * .2
        self.feature_scale.copy_(spread.clamp_min(floor))
        x = self.standardize(features).flatten(1).double()
        gram = (x @ x.T) / x.size(1)
        gram.diagonal().add_(ridge)
        return (x.T @ torch.linalg.solve(gram, target.double()) / x.size(1)).float()

    def encode_aperture(self, features):
        x = self.standardize(features)
        traces = self.trace_encoder(x)
        shots = self.shot_encoder(traces.flatten(2))
        return x, self.aperture_encoder(shots.flatten(1))


class EchoReconstructor(ApertureEncoder):
    """Compact-object affine geometry with a learned local shape correction."""

    architecture = "echo_affine_local_sdf_v1"

    def __init__(self, feature_shape, grid_shape, hidden=128, local_size=7):
        super().__init__(feature_shape, grid_shape, hidden)
        self.config["local_size"] = self.local_size = local_size
        self.register_buffer("moment_mean", torch.zeros(9))
        self.register_buffer("moment_scale", torch.ones(9))
        self.register_buffer("linear_moments", torch.zeros(math.prod(feature_shape), 9))
        self.geometry = nn.Linear(hidden, 9)
        self.local_field = nn.Linear(hidden, local_size ** 3)
        for head in (self.geometry, self.local_field):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def fit_statistics(self, features, moments, ridge):
        self.moment_mean.copy_(moments.mean(0))
        self.moment_scale.copy_(moments.std(0, unbiased=False).clamp_min(
            moments.new_tensor([1., 1., 1., .08, .08, .08, .08, .08, .08])))
        target = (moments - self.moment_mean) / self.moment_scale
        self.linear_moments.copy_(self.fit_readout(features, target, ridge))

    def encode(self, features):
        x, hidden = self.encode_aperture(features)
        standardized = x.flatten(1) @ self.linear_moments + self.geometry(hidden)
        params = standardized * self.moment_scale + self.moment_mean
        center = params[:, :3]
        radius = params[:, 3].clamp(-1., math.log(max(self.config["grid_shape"]))).exp()
        lower = affine_shape(params)
        inverse = torch.linalg.solve_triangular(lower, torch.eye(3, device=x.device).expand_as(lower), upper=False)
        correction = .55 * self.local_field(hidden).tanh().reshape(-1, 1, self.local_size, self.local_size,
                                                                  self.local_size)
        return standardized, center, radius, inverse, correction

    @staticmethod
    def decode(state, query):
        _, center, radius, inverse, correction = state
        delta = query - center[:, None]
        local = delta @ inverse.transpose(1, 2)
        # Grid axes are x,y,z. PyTorch orders sample coordinates from the last spatial axis to the first.
        sample = (local / (1.65 * radius[:, None, None])).flip(-1)[:, :, None, None]
        residual = F.grid_sample(correction, sample, mode="bilinear", padding_mode="border", align_corners=True)
        return base_distance(delta, local, radius) + radius[:, None] * residual[:, 0, :, 0, 0]


class VolumeReconstructor(ApertureEncoder):
    """Refine an echo-conditioned spatial feature grid into an unrestricted full-resolution field."""

    architecture = "echo_volume_sdf_v1"
    band = 4.

    def __init__(self, feature_shape, grid_shape, hidden=128):
        super().__init__(feature_shape, grid_shape, hidden)
        self.seed_shape = tuple(min(n, max(2, min(9, (n + 3) // 4))) for n in grid_shape)
        self.middle_shape = tuple((n + 1) // 2 for n in grid_shape)
        n_seed = math.prod(self.seed_shape)
        self.register_buffer("spatial_mean", torch.zeros(n_seed))
        self.register_buffer("linear_spatial", torch.zeros(math.prod(feature_shape), n_seed))
        self.seed_features = nn.Linear(hidden, 12 * n_seed)
        self.coarse = nn.Sequential(nn.Conv3d(13, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU())
        self.middle = nn.Sequential(nn.Conv3d(24, 12, 3, padding=1), nn.GroupNorm(3, 12), nn.SiLU())
        self.fine = nn.Sequential(nn.Conv3d(12, 12, 3, padding=1, groups=12), nn.GroupNorm(3, 12), nn.SiLU(),
                                  nn.Conv3d(12, 1, 1))
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    @torch.no_grad()
    def fit_statistics(self, features, sdf, ridge):
        coarse = F.interpolate(sdf[:, None], self.seed_shape, mode="trilinear", align_corners=True)
        targets = (coarse.flatten(1) / self.band).asinh()
        self.spatial_mean.copy_(targets.mean(0))
        self.linear_spatial.copy_(self.fit_readout(features, targets - self.spatial_mean, ridge))

    def encode(self, features):
        x, hidden = self.encode_aperture(features)
        prior = (x.flatten(1) @ self.linear_spatial + self.spatial_mean).reshape(-1, 1, *self.seed_shape)
        seed = self.seed_features(hidden).reshape(-1, 12, *self.seed_shape)
        coarse = self.coarse(torch.cat((prior, seed), dim=1))
        mid = self.middle(F.interpolate(coarse, self.middle_shape, mode="trilinear", align_corners=True))
        shape = self.config["grid_shape"]
        baseline = F.interpolate(prior, shape, mode="trilinear", align_corners=True)
        residual = self.fine(F.interpolate(mid, shape, mode="trilinear", align_corners=True))
        return self.band * (baseline + residual).clamp(-4, 4).sinh().squeeze(1)

    @staticmethod
    def decode(field, query):
        # The volume decoder predicts native grid cells directly.
        x, y, z = query.long().unbind(-1)
        index = (x * field.size(2) + y) * field.size(3) + z
        return field.flatten(1).gather(1, index.expand(field.size(0), -1))

    def loss(self, features, target):
        pred = self.encode(features)
        truth = target.flatten(1)
        pred = pred.flatten(1)
        occupied = truth < 0
        boundary = truth.abs() <= 2
        error = F.smooth_l1_loss((pred / self.band).asinh(), (truth / self.band).asinh(),
                                 reduction="none", beta=.12)
        near = (error * boundary).sum(1) / boundary.sum(1).clamp_min(1)
        logits = -pred / .75
        positive = (F.softplus(-logits) * occupied).sum(1) / occupied.sum(1).clamp_min(1)
        negative = (F.softplus(logits) * ~occupied).sum(1) / (~occupied).sum(1).clamp_min(1)
        probability = logits.sigmoid()
        dice = (2 * (probability * occupied).sum(1) + 1) / (probability.sum(1) + occupied.sum(1) + 1)
        return (error.mean(1) + near + .20 * (positive + negative) + .50 * (1 - dice)).mean()


def load_split(dataset, split, device):
    indices = dataset.splits[split]
    if not indices:
        raise ValueError(f"The {split} split is empty")
    signals, targets = [], []
    for i in indices:
        grid, waves = dataset[i]
        signals.append(waves)
        targets.append((grid[..., 2] - dataset.sdf_zero) * dataset.sdf_scale)
    waves = torch.stack(signals)
    features = torch.cat([trace_features(part.to(device)) for part in waves.split(64)])
    sdf = torch.stack(targets).to(device)
    return features, sdf, [dataset.scene_ids[i] for i in indices]


def training_loss(model, features, targets, moments, shape_targets, shape_weights, xyz, queries, field_weight):
    state = model.encode(features)
    standardized, _, _, _, correction = state
    target_params = (moments - model.moment_mean) / model.moment_scale
    moment_loss = F.smooth_l1_loss(standardized, target_params, beta=.5)
    full = targets.flatten(1)
    # Allocate samples separately to occupied voxels, both sides of the boundary, and the whole volume.
    n = queries // 3
    occupied = torch.multinomial((full < 0).float(), n, replacement=True)
    distance = full.abs()
    surface = (distance <= distance.amin(1, keepdim=True).clamp_min(1.5)).float()
    near = torch.multinomial(surface, n, replacement=True)
    uniform = torch.randint(full.size(1), (full.size(0), queries - 2*n), device=full.device)
    indices = torch.cat((occupied, near, uniform), dim=1)
    truth = full.gather(1, indices)
    pred = model.decode(state, xyz[indices])
    radius = moments[:, 3].exp()[:, None]
    field = F.smooth_l1_loss((pred / radius).clamp(-2, 2), (truth / radius).clamp(-2, 2), beta=.15)
    signs = F.binary_cross_entropy_with_logits(-pred / (.35 + .12 * radius), (truth < 0).float())
    shape_error = F.smooth_l1_loss(correction.flatten(1), shape_targets, reduction="none", beta=.10)
    local_loss = (shape_error * shape_weights).sum() / shape_weights.sum().clamp_min(1)
    smooth = sum(correction.diff(dim=i).square().mean() for i in (2, 3, 4))
    return moment_loss + local_loss + field_weight * (field + .25 * signs) + .015 * smooth


@torch.inference_mode()
def score(model, features, truth, xyz, scene_ids, batch_size, split="test"):
    model.eval()
    rows = []
    for start in range(0, len(features), batch_size):
        state = model.encode(features[start:start + batch_size])
        labels = truth[start:start + batch_size].flatten(1)
        mask = labels < 0
        target_centers = mask.float() @ xyz / mask.sum(1, keepdim=True).clamp_min(1)
        pred_sums = torch.zeros_like(target_centers)
        prediction_masks = []
        totals = torch.zeros((len(labels), 4), device=xyz.device, dtype=torch.float64)
        for i in range(0, xyz.size(0), 8192):
            query = xyz[i:i+8192]
            pred = model.decode(state, query[None])
            target = labels[:, i:i+len(query)]
            a, b = pred < 0, target < 0
            pred_sums += a.float() @ query
            if split == "test":
                prediction_masks.append(a)
            totals[:, 0] += (a & b).sum(1)
            totals[:, 1] += a.sum(1)
            totals[:, 2] += b.sum(1)
            totals[:, 3] += (pred - target).abs().sum(1)
        errors = (pred_sums / totals[:, 1:2].clamp_min(1) - target_centers).norm(dim=-1).cpu().tolist()
        if split == "test":
            shape = model.config["grid_shape"]
            predictions = torch.cat(prediction_masks, dim=1).reshape(-1, *shape).cpu().numpy()
            actual = mask.reshape(-1, *shape).cpu().numpy()
        for j, (intersection, predicted, occupied, error) in enumerate(totals.cpu().tolist()):
            union = int(predicted + occupied - intersection)
            row = dict(scene_id=scene_ids[start+j], occupancy_iou=intersection / union if union else 1.,
                       occupied_intersection=int(intersection), occupied_union=union,
                       occupied_target=int(occupied), occupied_predicted=int(predicted),
                       sdf_mae_cells=error / xyz.size(0), center_error_cells=errors[j] if occupied and predicted else None)
            if split == "test":
                row.update(component_counts(predictions[j], actual[j]))
                row["physical_instances"] = component_counts(predictions[j], actual[j], connectivity=26)
            rows.append(row)
    total_union = sum(row["occupied_union"] for row in rows)
    center_errors = [row["center_error_cells"] for row in rows if row["center_error_cells"] is not None]
    summary = {
        f"{split}_scenes": len(rows), "grid_shape": model.config["grid_shape"],
        "mean_scene_iou": sum(row["occupancy_iou"] for row in rows) / len(rows),
        "pooled_iou": sum(row["occupied_intersection"] for row in rows) / total_union if total_union else 1.,
        "mean_center_error_cells": sum(center_errors) / len(center_errors) if center_errors else None,
        "mean_sdf_mae_cells": sum(row["sdf_mae_cells"] for row in rows) / len(rows),
    }
    if split == "test":
        def counts(group):
            face = summarize_components(group)
            physical = summarize_components([r["physical_instances"] for r in group])
            return dict(**face, component_connectivity=6,
                        component_count_accuracy=face["correct_component_count_scenes"]/len(group),
                        physical_instances=dict(connectivity=26, **physical,
                                                count_accuracy=physical["correct_component_count_scenes"]/len(group)))

        summary.update(counts(rows))
        summary["component_iou_thresholds"] = [.25, .50]
        for name, key in [("by_target_components", None), ("by_target_instances", "physical_instances")]:
            groups = {}
            for row in rows:
                group_count = (row[key] if key else row)["target_components"]
                groups.setdefault(group_count, []).append(row)
            summary[name] = {
                str(k): {"scenes": len(group), "mean_scene_iou": sum(r["occupancy_iou"] for r in group)/len(group),
                         **counts(group)} for k, group in sorted(groups.items())
            }
    return {"scenes": rows, "summary": summary}


def save_checkpoint(path, model, dataset, epoch, summary, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(architecture=model.architecture, config=model.config, weights=model.state_dict(),
                      waveform_mode=dataset.waveform_mode, preprocessing_id=dataset.preprocessing_id,
                      training_protocol=dataset.training_protocol, epoch=epoch, validation=summary, seed=seed)
    with NamedTemporaryFile(dir=path.parent, prefix=".echo-", suffix=".pt", delete=False) as temp:
        pending = Path(temp.name)
    try:
        torch.save(checkpoint, pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def train(args):
    if min(args.batch_size, args.epochs, args.val_every, args.patience) < 1 or args.queries < 3 or args.ridge <= 0:
        raise ValueError("Positive batch/epoch/validation/patience/ridge settings and at least three queries are required")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    archive = open_archive(args.data, labels=True)
    features, sdf, _ = load_split(archive, "train", device)
    val_x, val_sdf, val_ids = load_split(archive, "validation", device)
    shape = sdf.shape[1:]
    xyz = coordinates(shape, device)
    if getattr(args, "decoder", "affine") == "volume":
        model = VolumeReconstructor(features.shape[1:], shape, hidden=args.hidden).to(device)
        model.fit_statistics(features, sdf, args.ridge)
    else:
        model = EchoReconstructor(features.shape[1:], shape, hidden=args.hidden).to(device)
        moments = torch.cat([object_moments(part, xyz) for part in sdf.split(32)])
        sampled = [local_training_targets(part, ms, model.local_size) for part, ms in zip(sdf.split(32), moments.split(32))]
        shape_targets, shape_weights = (torch.cat(values) for values in zip(*sampled))
        model.fit_statistics(features, moments, args.ridge)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.08)
    initial = score(model, val_x, val_sdf, xyz, val_ids, args.batch_size, "validation")["summary"]
    best, best_epoch = initial["mean_scene_iou"], 0
    save_checkpoint(args.checkpoint, model, archive, 0, initial, args.seed)
    print(json.dumps(dict(epoch=0, validation=initial)), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids in torch.randperm(len(features), device=device).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, VolumeReconstructor):
                loss = model.loss(features[ids], sdf[ids])
            else:
                loss = training_loss(model, features[ids], sdf[ids], moments[ids], shape_targets[ids], shape_weights[ids],
                                     xyz, args.queries, min(1., epoch / 12))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
        schedule.step()
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            summary = score(model, val_x, val_sdf, xyz, val_ids, args.batch_size, "validation")["summary"]
            value = summary["mean_scene_iou"]
            print(json.dumps(dict(epoch=epoch, validation=summary)), flush=True)
            if value > best:
                best, best_epoch = value, epoch
                save_checkpoint(args.checkpoint, model, archive, epoch, summary, args.seed)
            if epoch - best_epoch >= args.patience:
                break
    print(json.dumps(dict(checkpoint=str(args.checkpoint), best_epoch=best_epoch, validation_mean_iou=best)))
    return best


def restore(path, dataset, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    models = {cls.architecture: cls for cls in (EchoReconstructor, VolumeReconstructor)}
    if checkpoint.get("architecture") not in models:
        raise ValueError("Expected an EchoReconstructor checkpoint")
    check_preprocessing(dataset, checkpoint["waveform_mode"], checkpoint["preprocessing_id"])
    model = models[checkpoint["architecture"]](**checkpoint["config"])
    model.load_state_dict(checkpoint["weights"])
    model.training_protocol = checkpoint["training_protocol"]
    return model.to(device).eval()


def evaluate(args):
    device = torch.device(args.device)
    archive = open_archive(args.data, labels=True)
    model = restore(args.checkpoint, archive, device)
    x, sdf, names = load_split(archive, "test", device)
    xyz = coordinates(sdf.shape[1:], device)
    report = {"measured": score(model, x, sdf, xyz, names, args.batch_size)}
    if args.ablate:
        # Suppress the measured voltage before feature extraction, including its absolute level.
        zeros = trace_features(torch.zeros((1, *archive.sensor_shape), device=device)).expand_as(x)
        report["zero_echo"] = score(model, zeros, sdf, xyz, names, args.batch_size)
        if len(x) > 1:
            report["shuffled_echo"] = score(model, x.roll(1, 0), sdf, xyz, names, args.batch_size)
            for i, row in enumerate(report["shuffled_echo"]["scenes"]):
                row["echo_scene_id"] = names[i-1]
    print(json.dumps(report, indent=2))
    return report


def measured_waveform(scene, protocol):
    """Apply the training calibration to recorded voltages before any float32 conversion."""
    scene = Path(scene)
    settings = protocol["settings"]
    expected = settings["acquisition"]
    metadata = json.loads((scene / "acquisition.json").read_text())
    if not expected or metadata != expected:
        raise ValueError("Measured acquisition settings do not match the checkpoint training configuration")
    shape = tuple(settings["sensor_shape"])
    beamforming = settings["sensor_storage"] == "antenna_direction,time"
    layout = metadata.get("sensor_layout")
    observed = _load_trace(scene / "sensor_samples.npy", beamforming, layout)
    reference = _load_trace(scene / "empty_sensor_samples.npy", beamforming, layout)
    if observed.shape != shape or reference.shape != shape:
        raise ValueError(f"Expected a measured and matched empty waveform with canonical shape {shape}")
    scales = protocol["normalization"]
    normalized = ((observed - reference - scales["sensor_offset"]) / scales["sensor_scale"]).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Measured echo exceeds the usable range of the fitted calibration")
    return torch.from_numpy(normalized)


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    scene_dir = getattr(args, "scene_dir", None)
    if scene_dir is None:
        archive = open_archive(args.data, labels=False)
        model = restore(args.checkpoint, archive, device)
        scene_id = args.scene_id
        if scene_id not in archive.scene_ids:
            raise ValueError(f"Provide an existing --scene-id from the archive; received {scene_id!r}")
        waveform = archive[archive.scene_ids.index(scene_id)]
    else:
        model = restore(args.checkpoint, None, device)
        scene_id = args.scene_id or Path(scene_dir).name
        waveform = measured_waveform(scene_dir, model.training_protocol)
    state = model.encode(trace_features(waveform.unsqueeze(0).to(device)))
    shape = model.config["grid_shape"]
    xyz = coordinates(shape, device)
    field = torch.cat([model.decode(state, part[None]).flatten().cpu() for part in xyz.split(8192)]).reshape(shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    acquisition = model.training_protocol.get("settings", {}).get("acquisition", {})
    spatial_metadata = {key: acquisition[key] for key in ("grid_spacing_m", "grid_origin_cells") if key in acquisition}
    with args.output.open("wb") as dest:
        np.savez_compressed(dest, sdf_cells=field.numpy(), occupied=(field < 0).numpy(),
                            scene_id=scene_id, **spatial_metadata)
    print(json.dumps(dict(scene_id=scene_id, output=str(args.output), occupied=int((field < 0).sum()))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("train", "evaluate", "predict"):
        p = actions.add_parser(name)
        if name == "predict":
            source = p.add_mutually_exclusive_group(required=True)
            source.add_argument("--data", type=Path, help="Calibrated archive for --scene-id")
            source.add_argument("--scene-dir", type=Path, help="Directory with acquisition.json, sensor_samples.npy "
                                "and matched empty_sensor_samples.npy")
        else:
            p.add_argument("--data", type=Path, required=True, help="Calibrated archive directory or dataset.hdf5")
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if name != "predict":
            p.add_argument("--batch-size", type=int, default=32)
        if name == "train":
            p.add_argument("--decoder", choices=("affine", "volume"), default="affine",
                           help="Affine compact-object prior or unrestricted full-volume decoder")
            p.add_argument("--epochs", type=int, default=240)
            p.add_argument("--val-every", type=int, default=4)
            p.add_argument("--patience", type=int, default=48, help="Epochs without higher validation mean IoU")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--hidden", type=int, default=128)
            p.add_argument("--ridge", type=float, default=.10)
            p.add_argument("--lr", type=float, default=.001)
            p.add_argument("--queries", type=int, default=768, help="Sampled queries per scene for affine training")
        elif name == "evaluate":
            p.add_argument("--ablate", action="store_true", help="Also score zeroed and mismatched test echoes")
        else:
            p.add_argument("--scene-id", help="Scene to predict from the archive, or optional output ID for raw input")
            p.add_argument("--output", type=Path, required=True, help="NPZ with physical SDF in cells and occupancy")
    args = parser.parse_args()
    return {"train": train, "evaluate": evaluate, "predict": predict}[args.action](args)


if __name__ == "__main__":
    main()
