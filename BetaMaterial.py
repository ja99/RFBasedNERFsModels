"""Heteroscedastic Gaussian material regression on calibrated broadside RF and registered depth."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import ndtri

from BetaGaussian import beta_gaussian_loss
from NeuralDepthMaterial import (
    MaterialNetwork,
    NeuralDepthMaterial,
    SEEDS,
    baseline_predictions,
    calibrated_subset,
    read_split,
    sensor_features,
)
from EchoMaterial import PARAMETERS


LOSSES = {"mse": None, "nll": 0.0, "beta_half": 0.5, "beta_one": 1.0}
Z90 = float(ndtri(0.95))


def objective(mean, variance, target, beta):
    if beta is None:
        return (target - mean).square().mean()
    return beta_gaussian_loss(mean, variance, target, beta)


class GaussianMaterialNetwork(MaterialNetwork):
    def __init__(self, shapes):
        super().__init__(shapes)
        old = self.material[-1]
        head = nn.Linear(old.in_features, 4)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
            head.weight[:2].copy_(old.weight)
            head.bias[:2].copy_(old.bias)
        self.material[-1] = head

    def forward(self, *groups):
        mean, logvar = self.normalized_output(*groups).chunk(2, -1)
        return mean, logvar.clamp(-8.0, 6.0).exp()


def train_member(x, y, val, target, beta, seed):
    torch.manual_seed(seed)
    net = GaussianMaterialNetwork([g.shape[1:] for g in x])
    net.calibrate(x, y)
    train_y = (y - net.target_mean) / net.target_scale
    val_y = (target - net.target_mean) / net.target_scale
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.025)
    best, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, 86):
        net.train()
        for idx in torch.randperm(len(y)).split(56):
            mean, variance = net(*(g[idx] for g in x))
            loss = objective(mean, variance, train_y[idx], beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 4.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            val_mean, _ = net(*val)
            score = (val_y - val_mean).square().mean().item()
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if epoch == 45:
            opt.param_groups[0]["lr"] *= 0.4
        if epoch - best_epoch >= 22:
            break
    net.load_state_dict(best_state)
    return net.eval(), best_epoch


@torch.inference_mode()
def physical_moments(nets, groups):
    means, variances = [], []
    for net in nets:
        mu, var = net(*groups)
        means.append((mu.double() * net.target_scale + net.target_mean).numpy())
        variances.append((var.double() * net.target_scale.double().square()).numpy())
    return np.mean(means, axis=0), np.mean(variances, axis=0) + np.var(means, axis=0)


def nll(mean, var, y):
    return 0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)


def calibrate_variance(mean, var, y):
    error2 = (y - mean) ** 2
    return error2.mean(0).clip(1e-15), (error2 / var).mean(0).clip(1e-12)


def fitted_variance(var, meta):
    if meta["loss"] == "mse":
        return np.broadcast_to(meta["constant_variance"], var.shape)
    return var * np.array(meta["variance_multiplier"])


class BetaMaterial:
    def __init__(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.meta = payload["meta"]
        if self.meta["model_type"] != "beta_material" or self.meta["version"] != 1:
            raise ValueError("Expected a beta material checkpoint")
        self.settings = self.meta["settings"]
        self.mode = "fusion"
        self.nets = []
        for state in payload["states"]:
            net = GaussianMaterialNetwork(self.meta["shapes"])
            net.load_state_dict(state)
            self.nets.append(net.eval())

    def predict(self, wave, depth, camera):
        if camera != self.meta["camera"]:
            raise ValueError("Camera calibration differs from fitted model")
        features = sensor_features(wave, depth, camera, self.settings)
        groups = [torch.from_numpy(g[None]) for g in features]
        mean, var = physical_moments(self.nets, groups)
        return mean[0], fitted_variance(var, self.meta)[0]

    def predict_files(self, scene, depth, camera):
        return self.predict(
            calibrated_subset(Path(scene), self.settings),
            np.load(depth, allow_pickle=False),
            json.loads(Path(camera).read_text()),
        )


class PreviousBlendUncertainty:
    """Keep the previous blend's mean and calibrate a separate beta model's uncertainty."""

    def __init__(self, beta_path, point_path, kernel_path, calibration):
        spec = json.loads(Path(calibration).read_text())
        if spec["version"] != 1:
            raise ValueError("Unsupported blend uncertainty calibration")
        paths = dict(beta=beta_path, point=point_path, kernel=kernel_path)
        for name, path in paths.items():
            if (
                hashlib.sha256(Path(path).read_bytes()).hexdigest()
                != spec["sha256"][name]
            ):
                raise ValueError(
                    f"Calibration belongs to a different {name} checkpoint"
                )
        self.beta = BetaMaterial(beta_path)
        self.blend = NeuralDepthMaterial(point_path, kernel_path)
        self.multiplier = np.asarray(spec["multiplier"])

    def predict(self, wave, depth, camera):
        mean = self.blend.predict(wave=wave, depth=depth, camera=camera)
        _, variance = self.beta.predict(wave, depth, camera)
        return mean, self.multiplier * variance

    def predict_files(self, scene, depth, camera):
        return self.predict(
            calibrated_subset(Path(scene), self.beta.settings),
            np.load(depth, allow_pickle=False),
            json.loads(Path(camera).read_text()),
        )


def fit(train, frames, output):
    camera = json.loads((frames / "camera.json").read_text())
    settings, ids, x, y, background = read_split(train, frames, "train", camera)
    _, val_ids, v, target, _ = read_split(train, frames, "validation", camera)
    if set(ids) & set(val_ids):
        raise ValueError("Train and validation IDs overlap")
    yval = target.numpy().astype(float)
    scores = {}
    output.mkdir(parents=True, exist_ok=True)
    for name, beta in LOSSES.items():
        nets, epochs = zip(
            *(train_member(x, y, v, target, beta, seed) for seed in SEEDS)
        )
        mean, var = physical_moments(nets, v)
        const, k = calibrate_variance(mean, var, yval)
        meta = dict(
            model_type="beta_material",
            version=1,
            settings=settings,
            camera=camera,
            shapes=[list(g.shape[1:]) for g in x],
            training_ids=ids,
            validation_ids=val_ids,
            loss=name,
            background=background.tolist(),
            seeds=list(SEEDS),
            epochs=list(epochs),
            constant_variance=const.tolist(),
            variance_multiplier=k.tolist(),
        )
        scores[name] = nll(mean, fitted_variance(var, meta), yval).mean(0).tolist()
        torch.save(
            dict(meta=meta, states=[n.state_dict() for n in nets]),
            output / f"{name}.pt",
        )
        print(name, epochs, scores[name], flush=True)
    selected = min((k for k in scores if k != "mse"), key=lambda k: np.mean(scores[k]))
    (output / "selection.json").write_text(
        json.dumps(dict(selected=selected, validation_nll=scores), indent=2) + "\n"
    )


def metrics(mu, var, truth):
    err = truth - mu
    sd = np.sqrt(var)
    return dict(
        mae=np.abs(err).mean(0).tolist(),
        nll=nll(mu, var, truth).mean(0).tolist(),
        coverage90=(np.abs(err) <= Z90 * sd).mean(0).tolist(),
        width90=(2 * Z90 * sd).mean(0).tolist(),
    )


def evaluate(train, train_frames, tests, output, old_point, kernel):
    selected = json.loads((output / "selection.json").read_text())["selected"]
    models = {name: BetaMaterial(output / f"{name}.pt") for name in LOSSES}
    old = NeuralDepthMaterial(old_point, kernel)

    def historical(path, frames, split):
        camera = json.loads((frames / "camera.json").read_text())
        settings, ids, x, y, _ = read_split(path, frames, split, camera)
        if settings != old.settings or camera != old.meta["camera"]:
            raise ValueError("Incompatible historical model calibration")
        with torch.inference_mode():
            mu = np.mean([net(*x).numpy() for net in old.models], axis=0)
        base = baseline_predictions(old.baseline, path, frames, split, ids, camera)
        w = np.asarray(old.meta["neural_weights"])
        return (
            ids,
            x,
            y.numpy().astype(float),
            {"previous_neural": mu, "previous_blend": w * mu + (1 - w) * base},
        )

    _, v, yv, hv = historical(train, train_frames, "validation")
    fixed = {k: np.mean((yv - v) ** 2, axis=0) for k, v in hv.items()}
    selected_model = models[selected]
    _, raw_variance = physical_moments(selected_model.nets, v)
    _, blend_scale = calibrate_variance(
        hv["previous_blend"], fitted_variance(raw_variance, selected_model.meta), yv
    )
    paths = dict(beta=output / f"{selected}.pt", point=old_point, kernel=kernel)
    blend_spec = dict(
        version=1,
        multiplier=blend_scale.tolist(),
        sha256={
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    )
    (output / "blend_variance.json").write_text(json.dumps(blend_spec, indent=2) + "\n")
    report = dict(selected=selected, parameters=list(PARAMETERS), cohorts={})
    for key, (path, frames) in tests.items():
        ids, groups, truth, hist = historical(path, frames, "test")
        if set(ids) & set(old.meta["training_ids"] + old.meta["validation_ids"]):
            raise ValueError("Held-out overlap")
        preds = {k: (v, np.broadcast_to(fixed[k], v.shape)) for k, v in hist.items()}
        raw_stats = {}
        for name, model in models.items():
            if (
                model.settings != old.settings
                or model.meta["training_ids"] != old.meta["training_ids"]
                or model.meta["validation_ids"] != old.meta["validation_ids"]
            ):
                raise ValueError("Benchmark splits or settings differ")
            mu, var = physical_moments(model.nets, groups)
            preds[name] = (mu, fitted_variance(var, model.meta))
            if name != "mse":
                raw_stats[name] = metrics(mu, var, truth)
            if name == selected:
                preds["same_mean_fixed_variance"] = (
                    mu,
                    np.broadcast_to(model.meta["constant_variance"], var.shape),
                )
        preds["previous_blend_beta_variance"] = (
            hist["previous_blend"],
            preds[selected][1] * blend_scale,
        )
        statistics = {
            name: metrics(mu, var, truth) for name, (mu, var) in preds.items()
        }
        mu, var = preds[selected]
        draws = np.random.default_rng(617901).integers(len(ids), size=(5000, len(ids)))
        deltas = {}
        for other in ("mse", "nll", "previous_blend", "same_mean_fixed_variance"):
            m, s = preds[other]
            comp = dict(
                mae=np.abs(mu - truth) - np.abs(m - truth),
                nll=nll(mu, var, truth) - nll(m, s, truth),
            )
            deltas[other] = {
                k: dict(
                    change=a.mean(0).tolist(),
                    ci95=np.quantile(
                        a[draws].mean(1), [0.025, 0.975], axis=0
                    ).T.tolist(),
                )
                for k, a in comp.items()
            }
        blend_mean, blend_var = preds["previous_blend_beta_variance"]
        effect = nll(blend_mean, blend_var, truth) - nll(
            blend_mean, preds["previous_blend"][1], truth
        )
        report["cohorts"][key] = dict(
            scenes=len(ids),
            calibrated=statistics,
            raw=raw_stats,
            paired_selected_minus=deltas,
            previous_blend_beta_minus_constant_nll=dict(
                change=effect.mean(0).tolist(),
                ci95=np.quantile(
                    effect[draws].mean(1), [0.025, 0.975], axis=0
                ).T.tolist(),
            ),
        )
        arrays = {"scene": np.asarray(ids), "truth": truth}
        for name, (mu, var) in preds.items():
            arrays[name + "_mean"] = mu
            arrays[name + "_var"] = var
        np.savez_compressed(output / f"{key}_predictions.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    study = sub.add_parser("study")
    for name in (
        "view",
        "frames",
        "independent",
        "independent-frames",
        "point",
        "kernel",
        "output",
    ):
        study.add_argument("--" + name, type=Path, required=True)
    study.add_argument("--score-only", action="store_true")
    infer = sub.add_parser("infer")
    for name in ("model", "scene", "depth", "camera"):
        infer.add_argument("--" + name, type=Path, required=True)
    for flag in ("point", "kernel", "blend-calibration"):
        infer.add_argument(
            "--" + flag,
            type=Path,
            help="Combine historical blend mean with calibrated beta variance",
        )
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.command == "infer":
        inputs = (a.point, a.kernel, a.blend_calibration)
        if any(inputs) and not all(inputs):
            infer.error("Supply --point, --kernel, and --blend-calibration together")
        predictor = (
            PreviousBlendUncertainty(a.model, *inputs)
            if all(inputs)
            else BetaMaterial(a.model)
        )
        mu, var = predictor.predict_files(a.scene, a.depth, a.camera)
        print(
            json.dumps(
                {
                    k: dict(
                        mean=m,
                        variance=v,
                        interval90=[m - Z90 * v**0.5, m + Z90 * v**0.5],
                    )
                    for k, m, v in zip(PARAMETERS, mu, var)
                },
                indent=2,
            )
        )
        return
    if not a.score_only:
        fit(a.view / "train", a.frames, a.output)
    evaluate(
        a.view / "train",
        a.frames,
        dict(
            original=(a.view / "test", a.frames),
            independent=(a.independent, a.independent_frames),
        ),
        a.output,
        a.point,
        a.kernel,
    )


if __name__ == "__main__":
    main()
