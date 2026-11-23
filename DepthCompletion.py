"""Complete the convex support of one visible object from registered depth, optionally fused with RF."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from DepthGeometry import constrain, iou, registered_camera
from DepthMaterial import calibrated_subset
from EchoKernel import (
    DIRECTIONS,
    _archive,
    _echo_features,
    _fit_projection,
    _mask,
    _project,
    _settings,
    _signal,
    _support,
    _target,
    occupancy_sdf,
)


SEEDS = (19, 73, 145)
WIDTH = 96


def surface_features(frame, optics):
    free, surface = optics.observations(frame)
    points = np.argwhere(surface).astype(np.float32)
    if not len(points):
        raise ValueError("Completion requires a visible surface with a positive depth return")
    center = points.mean(0)
    directions = (points - center) @ DIRECTIONS.T
    visible = directions.max(0) + center @ DIRECTIONS.T
    # The support, spread and skew of the visible point cloud describe its front surface.
    # Retain location and image-edge coverage to represent partly clipped silhouettes.
    foreground = frame > 0
    boundary = [edge.mean() for edge in (foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1])]
    descriptor = np.concatenate(
        (
            center,
            directions.max(0),
            directions.std(0),
            np.cbrt((directions**3).mean(0)),
            [len(points), foreground.sum()],
            boundary,
        )
    )
    return descriptor.astype("f4"), visible, free, surface


class SupportCompletion(nn.Module):
    def __init__(self, depth_dim, rf_dim):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Linear(depth_dim, WIDTH), nn.LayerNorm(WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.GELU()
        )
        self.rf_encoder = (
            nn.Sequential(
                nn.Linear(rf_dim, WIDTH), nn.LayerNorm(WIDTH), nn.GELU(), nn.Linear(WIDTH, WIDTH), nn.GELU()
            )
            if rf_dim
            else None
        )
        n = WIDTH * (3 if rf_dim else 1)
        self.head = nn.Sequential(
            nn.Linear(n, WIDTH), nn.GELU(), nn.Dropout(0.08), nn.Linear(WIDTH, len(DIRECTIONS))
        )

    def forward(self, depth, rf):
        d = self.depth_encoder(depth)
        if self.rf_encoder is not None:
            r = self.rf_encoder(rf)
            d = torch.cat((d, r, d * r), dim=-1)
        return self.head(d)


def load_split(archive, split, frames, camera):
    settings = _settings(archive)
    optics = registered_camera(settings["acquisition"], settings["grid_shape"], camera)
    ids, dep, rf, supports, labels, surfaces, empty, truth = [], [], [], [], [], [], [], []
    for i in archive["splits/" + split][:]:
        scene = archive["scene_ids"].asstr()[i]
        z = np.load(frames / scene / "depth.npy", allow_pickle=False)
        features, visible, free, solid = surface_features(z, optics)
        target = _target(archive, i) < 0
        if np.any(solid & ~target) or np.any(free & target):
            raise ValueError("Depth observations contradict RF voxel geometry")
        ids.append(scene)
        dep.append(features)
        rf.append(_echo_features(_signal(archive, i, settings)))
        supports.append(visible)
        labels.append(np.maximum(0, _support(target) - visible))
        surfaces.append(solid)
        empty.append(free)
        truth.append(target)
    return dict(
        ids=ids,
        depth=np.stack(dep),
        rf=[np.stack(g) for g in zip(*rf)],
        visible=np.stack(supports),
        y=np.stack(labels),
        free=empty,
        surface=surfaces,
        truth=truth,
    )


def standardization(x):
    return x.mean(0), np.maximum(x.std(0), 1e-4)


def scaled(x, mean, scale):
    return torch.from_numpy(np.clip((x - mean) / scale, -8, 8).astype("f4"))


def train_member(d, r, y, dv, rv, yv, seed):
    torch.manual_seed(seed)
    net = SupportCompletion(d.shape[1], r.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=160, eta_min=0.0001)
    best, params, epoch_best = np.inf, None, None
    for epoch in range(160):
        net.train()
        for idx in torch.randperm(len(d)).split(64):
            pred = net(d[idx], r[idx])
            loss = nn.functional.smooth_l1_loss(pred, y[idx], beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            net.eval()
            with torch.inference_mode():
                error = float((net(dv, rv) - yv).abs().mean())
            if error < best:
                best = error
                params = copy.deepcopy(net.state_dict())
                epoch_best = epoch + 1
    net.load_state_dict(params)
    net.eval()
    return net, epoch_best


def decoded(visible, predicted, pad, free, surface):
    h = visible + np.maximum(predicted, 0.0)
    proposal = _mask(h, surface.shape, pad)
    return constrain(proposal, free, surface)


def fit(t, v, settings, camera, mode, path):
    df = standardization(t["depth"])
    train_d, val_d = (scaled(a["depth"], *df) for a in (t, v))
    transforms = []
    if mode == "fusion":
        rng = np.random.default_rng(221)
        transforms = [_fit_projection(g, j == 2, rng) for j, g in enumerate(t["rf"])]
        rt, rv = (np.concatenate(_project(a["rf"], transforms), axis=1) for a in (t, v))
        rf_norm = standardization(rt)
        train_r, val_r = scaled(rt, *rf_norm), scaled(rv, *rf_norm)
    else:
        rf_norm = (np.zeros(0, "f4"), np.ones(0, "f4"))
        train_r, val_r = (torch.zeros(len(a["ids"]), 0) for a in (t, v))
    # One common scale keeps the loss in physical support-cell units across directions.
    y_mean = t["y"].mean(0)
    y_scale = max(float(t["y"].std()), 0.01)
    train_y, val_y = (torch.from_numpy((a["y"] - y_mean) / y_scale) for a in (t, v))
    nets, epochs = zip(
        *(train_member(train_d, train_r, train_y, val_d, val_r, val_y, seed) for seed in SEEDS)
    )
    with torch.inference_mode():
        pred = torch.stack([model(val_d, val_r) for model in nets]).mean(0).numpy() * y_scale + y_mean
    choices = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    scores = [
        np.mean(
            [
                iou(decoded(h, ext, pad, f, p), label)
                for h, ext, f, p, label in zip(v["visible"], pred, v["free"], v["surface"], v["truth"])
            ]
        )
        for pad in choices
    ]
    select = int(np.argmax(scores))

    def pack(arrays):
        return [torch.from_numpy(array) for array in arrays]

    store = dict(
        version=1,
        mode=mode,
        settings=settings,
        camera=camera,
        d_norm=pack(df),
        r_norm=pack(rf_norm),
        transforms=[pack(block) for block in transforms],
        y_mean=torch.from_numpy(y_mean),
        y_scale=y_scale,
        pad=choices[select],
        epochs=list(epochs),
        state=[model.state_dict() for model in nets],
        fit_scene_ids=t["ids"],
        validation_scene_ids=v["ids"],
        validation_iou=float(scores[select]),
    )
    torch.save(store, path)
    return dict(
        mode=mode,
        validation_iou=float(scores[select]),
        offset=choices[select],
        epochs=epochs,
        parameters_per_member=sum(p.numel() for p in nets[0].parameters()),
    )


class Completion:
    def __init__(self, path, simulator=None):
        s = torch.load(path, weights_only=True, map_location="cpu")
        if s.get("version") != 1 or s.get("mode") not in ("depth", "fusion") or "state" not in s:
            raise ValueError("Unsupported checkpoint version")
        self.meta = s
        self.settings = s["settings"]
        self.optics = registered_camera(
            self.settings["acquisition"], self.settings["grid_shape"], s["camera"], simulator
        )
        self.d_norm, self.r_norm = ([v.numpy() for v in s[k]] for k in ("d_norm", "r_norm"))
        self.transforms = [[v.numpy() for v in block] for block in s["transforms"]]
        self.nets = []
        for state in s["state"]:
            net = SupportCompletion(len(self.d_norm[0]), len(self.r_norm[0]))
            net.load_state_dict(state)
            self.nets.append(net.eval())

    def predict(self, depth, camera, wave=None, tolerance_m=0.0):
        """Predict a full Boolean mask from one positive-return frame and calibrated RF input."""
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("Depth tolerance must be finite and nonnegative")
        if camera != self.meta["camera"]:
            raise ValueError("Depth calibration differs from training")
        features, visible, free, surface = surface_features(depth, self.optics)
        d = scaled(features[None], *self.d_norm)
        if self.transforms:
            if np.shape(wave) != tuple(self.settings["sensor_shape"]):
                raise ValueError("Expected calibrated training receiver, beam and time axes")
            raw = [g[None] for g in _echo_features(wave)]
            r = scaled(np.concatenate(_project(raw, self.transforms), axis=1), *self.r_norm)
        else:
            r = torch.zeros(1, 0)
        with torch.inference_mode():
            y = torch.stack([net(d, r)[0] for net in self.nets]).mean(0).numpy()
        extension = y * self.meta["y_scale"] + self.meta["y_mean"].numpy()
        if tolerance_m:
            free, surface = self.optics.observations(depth, tolerance_m=tolerance_m)
        return decoded(visible, extension, self.meta["pad"], free, surface)

    def predict_files(self, depth, camera, rf_scene=None, tolerance_m=0.0):
        wave = None
        if self.transforms:
            if rf_scene is None:
                raise ValueError("The fused geometry checkpoint requires an RF recording")
            wave = calibrated_subset(Path(rf_scene), self.settings)
        return self.predict(
            np.load(depth, allow_pickle=False), json.loads(Path(camera).read_text()), wave, tolerance_m
        )


def train(archive_path, frames, output):
    frames, output = Path(frames), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    camera = json.loads((frames / "camera.json").read_text())
    with _archive(archive_path) as archive:
        settings = _settings(archive)
        training = load_split(archive, "train", frames, camera)
        validation = load_split(archive, "validation", frames, camera)
    if set(training["ids"]) & set(validation["ids"]):
        raise ValueError("Training and validation contain overlapping scenes")
    rows = [
        fit(training, validation, settings, camera, mode, output / (mode + ".pt"))
        for mode in ("depth", "fusion")
    ]
    (output / "validation.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fitter = sub.add_parser("fit")
    for name in ("archive", "frames", "output"):
        fitter.add_argument(name, type=Path)
    infer = sub.add_parser("infer")
    for name in ("checkpoint", "depth", "camera", "output"):
        infer.add_argument(name, type=Path)
    infer.add_argument("--rf-scene", type=Path)
    infer.add_argument("--depth-tolerance-mm", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "fit":
        print(json.dumps(train(args.archive, args.frames, args.output), indent=2))
    else:
        input_paths = [args.checkpoint, args.depth, args.camera]
        if args.rf_scene:
            input_paths.extend(
                args.rf_scene / name
                for name in ("sensor_samples.npy", "empty_sensor_samples.npy", "acquisition.json")
            )
        if args.output.resolve() in {p.resolve() for p in input_paths}:
            parser.error("Choose a separate output path")
        model = Completion(args.checkpoint)
        mask = model.predict_files(args.depth, args.camera, args.rf_scene, args.depth_tolerance_mm / 1000.0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as handle:
            np.savez_compressed(handle, occupied=mask, sdf_cells=occupancy_sdf(mask))
