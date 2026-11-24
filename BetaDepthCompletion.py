"""Predict means and calibrated marginal variances of the 98 geometry support extensions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from BetaGaussian import beta_gaussian_loss
from DepthCompletion import (
    DIRECTIONS,
    SEEDS,
    WIDTH,
    SupportCompletion,
    _archive,
    _echo_features,
    _project,
    _settings,
    calibrated_subset,
    decoded,
    iou,
    load_split,
    registered_camera,
    scaled,
    surface_features,
)


FLOOR_SD = 0.20  # Voxel-cell units; sub-voxel resolution floor shared by all variance forecasts.
Z90 = 1.6448536269514722
ARMS = {"mse": None, "beta0": 0.0, "beta05": 0.5, "beta1": 1.0}


class DistributionNet(SupportCompletion):
    def __init__(self, nd, nr):
        super().__init__(nd, nr)
        self.variance = nn.Linear(WIDTH, len(DIRECTIONS))
        nn.init.zeros_(self.variance.weight)
        nn.init.zeros_(self.variance.bias)

    def forward(self, depth, rf):
        d, r = self.depth_encoder(depth), self.rf_encoder(rf)
        h = self.head[:-1](torch.cat((d, r, d * r), dim=-1))
        variance = nn.functional.softplus(self.variance(h)) + 0.005
        return self.head[-1](h), variance


def beta_loss(mean, variance, truth, beta):
    """Beta-NLL on normalized targets with the variance weight detached."""
    if beta is None:
        return (mean - truth).square().mean()
    return beta_gaussian_loss(mean, variance, truth, beta)


def inputs(data, meta):
    projected = np.concatenate(
        _project(data["rf"], [[a.numpy() for a in g] for g in meta["transforms"]]), axis=1
    )
    return (
        scaled(data["depth"], *(a.numpy() for a in meta["d_norm"])),
        scaled(projected, *(a.numpy() for a in meta["r_norm"])),
    )


def train_member(seed, beta, x, y, xv, yv):
    torch.manual_seed(seed)
    net = DistributionNet(x[0].shape[1], x[1].shape[1])
    epochs = 120
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0001)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        for idx in torch.randperm(len(y)).split(64):
            mu, var = net(x[0][idx], x[1][idx])
            opt.zero_grad()
            beta_loss(mu, var, y[idx], beta).backward()
            opt.step()
        scheduler.step()
        if epoch % 5 == 0:
            net.eval()
            with torch.inference_mode():
                mu, var = net(*xv)
                score = float((mu - yv).square().mean() if beta is None else beta_loss(mu, var, yv, 0.0))
            if score < best:
                best, state, best_epoch = score, copy.deepcopy(net.state_dict()), epoch
    net.load_state_dict(state)
    return net.eval(), best_epoch


@torch.inference_mode()
def moments(nets, x, meta):
    mus, variances = zip(*(net(*x) for net in nets))
    mu = torch.stack(mus)
    # Total marginal variance of an equally weighted ensemble, converted to voxel-cell units.
    var = (torch.stack(variances).mean(0) + mu.var(0, unbiased=False)) * meta["y_scale"] ** 2
    return mu.mean(0).numpy() * meta["y_scale"] + meta["y_mean"].numpy(), var.numpy()


def gaussian_nll(truth, mu, var):
    return 0.5 * (np.log(2 * np.pi * var) + (truth - mu) ** 2 / var)


def fit(train_archive, frames, point_checkpoint, output):
    from scipy.optimize import minimize_scalar

    old = torch.load(point_checkpoint, map_location="cpu", weights_only=True)
    meta = {
        k: old[k]
        for k in (
            "d_norm",
            "r_norm",
            "transforms",
            "y_mean",
            "y_scale",
            "camera",
            "settings",
            "fit_scene_ids",
            "validation_scene_ids",
        )
    }
    camera = json.loads((frames / "camera.json").read_text())
    with _archive(train_archive) as arc:
        if _settings(arc) != meta["settings"] or camera != meta["camera"]:
            raise ValueError("Input protocol differs from fitted RF feature projection")
        t, v = (load_split(arc, s, frames, camera) for s in ("train", "validation"))
    if (
        t["ids"] != meta["fit_scene_ids"]
        or v["ids"] != meta["validation_scene_ids"]
        or set(t["ids"]) & set(v["ids"])
    ):
        raise ValueError("Input split differs from fitted RF feature projection")
    x, xv = inputs(t, meta), inputs(v, meta)
    y, yv = (torch.from_numpy((data["y"] - meta["y_mean"].numpy()) / meta["y_scale"]) for data in (t, v))
    # Alternating validation IDs, shuffled with fixed seed to avoid any generation-order effect.
    order = np.random.default_rng(7847).permutation(len(yv))
    selection, calibration = np.array_split(order, 2)
    rows = {}
    for name, beta in ARMS.items():
        members, epochs = zip(
            *(train_member(s, beta, x, y, [a[selection] for a in xv], yv[selection]) for s in SEEDS)
        )
        mu, raw = moments(members, xv, meta)
        sq = (mu - v["y"]) ** 2
        const = np.maximum(sq[calibration].mean(0), FLOOR_SD**2)
        factor = float(
            np.exp(
                minimize_scalar(
                    lambda z: gaussian_nll(
                        v["y"][calibration],
                        mu[calibration],
                        np.maximum(raw[calibration] * np.exp(z), FLOOR_SD**2),
                    ).mean(),
                    method="bounded",
                    bounds=(-6, 6),
                ).x
            )
        )
        var = np.broadcast_to(const, raw.shape) if beta is None else np.maximum(raw * factor, FLOOR_SD**2)
        nll = float(gaussian_nll(v["y"][selection], mu[selection], var[selection]).mean())

        def overlap(pad):
            return float(
                np.mean(
                    [
                        iou(
                            decoded(v["visible"][i], mu[i], pad, v["free"][i], v["surface"][i]), v["truth"][i]
                        )
                        for i in selection
                    ]
                )
            )

        pads = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
        ious = [overlap(p) for p in pads]
        j = int(np.argmax(ious))
        netmeta = dict(
            meta,
            version=1,
            model_type="beta_support",
            beta=beta,
            scale_factor=factor,
            constant_variance=torch.from_numpy(const),
            pad=pads[j],
            state=[m.state_dict() for m in members],
        )
        torch.save(netmeta, output / (name + ".pt"))
        rows[name] = dict(
            beta=beta,
            validation_nll=nll,
            validation_iou=ious[j],
            pad=pads[j],
            epochs=list(epochs),
            scale_factor=factor,
        )
        print(name, rows[name], flush=True)
    selected = min((k for k in ARMS if k != "mse"), key=lambda k: rows[k]["validation_nll"])
    result = dict(
        selected=selected,
        seeds=list(SEEDS),
        methods=rows,
        selection_ids=[v["ids"][i] for i in selection],
        calibration_ids=[v["ids"][i] for i in calibration],
        variance_floor_cells2=FLOOR_SD**2,
    )
    attach_incumbent(output, selected, old, v, calibration)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def restore_point(meta, states):
    nets = []
    for state in states:
        net = SupportCompletion(len(meta["d_norm"][0]), len(meta["r_norm"][0]))
        net.load_state_dict(state)
        nets.append(net.eval())
    return nets


@torch.inference_mode()
def point_mean(nets, x, meta):
    return torch.stack([net(*x) for net in nets]).mean(0).numpy() * meta["y_scale"] + meta["y_mean"].numpy()


def attach_incumbent(output, winner, old, valid, calibration):
    """Keep the incumbent point estimate and recalibrate the selected uncertainty around that mean."""
    from scipy.optimize import minimize_scalar

    source = BetaCompletion(output / (winner + ".pt"))
    m = source.meta
    x = inputs(valid, m)
    mu = point_mean(restore_point(m, old["state"]), x, m)
    _, raw = moments(source.nets, x, m)
    m["scale_factor"] = float(
        np.exp(
            minimize_scalar(
                lambda z: gaussian_nll(
                    valid["y"][calibration],
                    mu[calibration],
                    np.maximum(raw[calibration] * np.exp(z), FLOOR_SD**2),
                ).mean(),
                method="bounded",
                bounds=(-6, 6),
            ).x
        )
    )
    m["constant_variance"] = torch.from_numpy(
        np.maximum(((valid["y"] - mu)[calibration] ** 2).mean(0), FLOOR_SD**2)
    )
    m.update(point_states=old["state"], pad=old["pad"])
    torch.save(m, output / "incumbent_mean.pt")


class BetaCompletion:
    def __init__(self, checkpoint):
        self.meta = m = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if m.get("model_type") != "beta_support" or m["version"] != 1:
            raise ValueError("Expected a beta support checkpoint")
        self.optics = registered_camera(
            m["settings"]["acquisition"], m["settings"]["grid_shape"], m["camera"]
        )
        self.nets = []
        for state in m["state"]:
            net = DistributionNet(len(m["d_norm"][0]), len(m["r_norm"][0]))
            net.load_state_dict(state)
            self.nets.append(net.eval())

        self.point_nets = restore_point(m, m.get("point_states", []))

    def forecast(self, x):
        mu, var = moments(self.nets, x, self.meta)
        if self.point_nets:
            mu = point_mean(self.point_nets, x, self.meta)
        if self.meta["beta"] is None:
            return mu, np.broadcast_to(self.meta["constant_variance"].numpy(), mu.shape)
        return mu, np.maximum(var * self.meta["scale_factor"], FLOOR_SD**2)

    def predict(self, wave, depth, camera):
        m = self.meta
        if camera != m["camera"] or tuple(np.shape(wave)) != tuple(m["settings"]["sensor_shape"]):
            raise ValueError("Input calibration differs from training")
        features, visible, free, solid = surface_features(depth, self.optics)
        x = inputs(dict(depth=features[None], rf=[g[None] for g in _echo_features(wave)]), m)
        mean, variance = self.forecast(x)
        return dict(
            directions=DIRECTIONS,
            visible_support_cells=visible,
            expected_support_cells=visible + mean[0],
            variance_support_cells2=variance[0],
            occupied=decoded(visible, mean[0], m["pad"], free, solid),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for command, names in [
        ("fit", ("archive", "frames", "point_checkpoint", "output")),
        ("infer", ("checkpoint", "rf_scene", "depth", "camera", "output")),
    ]:
        p = sub.add_parser(command)
        for arg in names:
            p.add_argument(arg, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "fit":
        args.output.mkdir(parents=True, exist_ok=True)
        fit(args.archive, args.frames, args.point_checkpoint, args.output)
    else:
        sources = [args.depth, args.checkpoint, args.camera]
        sources.extend(
            args.rf_scene / name
            for name in ("sensor_samples.npy", "empty_sensor_samples.npy", "acquisition.json")
        )
        if args.output.resolve() in {p.resolve() for p in sources}:
            parser.error("Choose a separate output path")
        model = BetaCompletion(args.checkpoint)
        prediction = model.predict(
            calibrated_subset(args.rf_scene, model.meta["settings"]),
            np.load(args.depth),
            json.loads(args.camera.read_text()),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as f:
            np.savez_compressed(f, **prediction)
