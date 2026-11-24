"""Geometry-conditioned material estimation for single-shot 4 × 4 RF and registered depth observations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from DepthMaterial import DepthMaterial, _depth_features, calibrated_subset
from EchoKernel import _archive, _settings, _signal
from EchoMaterial import PARAMETERS, _features, _homogeneous, _label


SEEDS = (829, 173, 293)
BLEND_WEIGHTS = np.linspace(0, 1, 5)


def sensor_features(wave, depth, camera, settings):
    if tuple(np.shape(wave)) != (16, 1, 128) or not np.isfinite(wave).all():
        raise ValueError("Expected calibrated 16-receiver, one-shot, 128-sample waveforms")
    acquisition = settings["acquisition"]
    positions = np.asarray(acquisition["antenna_positions_cells"])
    yz = positions[:, 1:]
    if (positions.shape != (16, 3) or np.ptp(positions[:, 0]) != 0 or
            any(len(np.unique(axis)) != 4 for axis in yz.T) or len(np.unique(yz, axis=0)) != 16):
        raise ValueError("A planar 4 × 4 array is required")
    # Keep position information while sharing a bank of learned temporal filters across receivers.
    order = np.lexsort((yz[:, 1], yz[:, 0]))
    traces = np.asarray(wave, dtype=np.float32)[order, 0]
    rms = np.maximum(np.sqrt(np.mean(traces**2, axis=1, keepdims=True)), 1e-8)
    spectrum = np.abs(np.fft.rfft(traces, axis=1)[:, :32]) / np.sqrt(traces.shape[1])
    channels = np.concatenate((traces / rms, np.log(np.maximum(spectrum, 1e-8)), np.log(rms)), axis=1)
    # Summaries provide signal scale and visible surface extent to the conditioning branch.
    depth_pixels, geometry = _depth_features(depth, camera)
    aggregate = _features(wave, acquisition)[-1]
    return (channels.astype("f4"), depth_pixels.reshape(2, 8, 8), np.r_[geometry, aggregate].astype("f4"))


def read_split(archive, frames, split, camera):
    with _archive(archive) as f:
        settings = _settings(f)
        rows = np.asarray(f["splits/" + split], dtype=int)
        ids = f["scene_ids"].asstr()[:][rows].tolist()
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Expected distinct scenes in {split}")
        inputs, targets, background = [], [], []
        for i, scene in zip(rows, ids):
            wave = _signal(f, i, settings)
            depth = np.load(frames / scene / "depth.npy", allow_pickle=False)
            inputs.append(sensor_features(wave, depth, camera, settings))
            target, bg = _label(f, i)
            targets.append(target)
            background.append(bg)
    groups = tuple(torch.from_numpy(np.stack(group)) for group in zip(*inputs))
    return settings, ids, groups, torch.tensor(np.stack(targets), dtype=torch.float32), _homogeneous(np.stack(background), "Background")


class MaterialNetwork(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        ports, image, summary = shapes
        self.temporal = nn.Sequential(nn.Linear(ports[-1], 40), nn.SiLU(), nn.Linear(40, 24))
        self.aperture = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(24, 16, 3, padding=1), nn.SiLU(), nn.Flatten(),
                                      nn.Linear(16 * 16, 40), nn.SiLU())
        self.surface = nn.Sequential(nn.Conv2d(image[0], 12, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                                     nn.Conv2d(12, 12, 3, padding=1), nn.SiLU(), nn.Flatten(),
                                     nn.Linear(12 * 4 * 4, 32), nn.SiLU())
        self.geometry = nn.Sequential(nn.Linear(32 + 32, 48), nn.SiLU())
        self.modulation = nn.Linear(48, 2 * 40)
        self.summary = nn.Sequential(nn.Linear(summary[-1] - 32, 64), nn.SiLU(), nn.Linear(64, 40))
        self.material = nn.Sequential(nn.Linear(40 + 40 + 48, 64), nn.SiLU(),
                                      nn.Dropout(.08), nn.Linear(64, 2))
        for i, shape in enumerate((ports[-1:], (2, 1, 1), summary)):
            self.register_buffer(f"mean{i}", torch.zeros(shape))
            self.register_buffer(f"scale{i}", torch.ones(shape))
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_scale", torch.ones(2))

    def calibrate(self, groups, y):
        for i, (x, axes) in enumerate(zip(groups, ((0, 1), (0, 2, 3), (0,)))):
            target_shape = getattr(self, f"mean{i}").shape
            getattr(self, f"mean{i}").copy_(x.mean(axes).reshape(target_shape))
            getattr(self, f"scale{i}").copy_(x.std(axes, unbiased=False).reshape(target_shape).clamp_min(1e-6))
        self.target_mean.copy_(y.mean(0))
        self.target_scale.copy_(y.std(0, unbiased=False).clamp_min(1e-6))

    def normalized_output(self, ports, image, summary):
        ports, image, summary = [((x - getattr(self, f"mean{i}")) / getattr(self, f"scale{i}")).clamp(-8, 8)
                                 for i, x in enumerate((ports, image, summary))]
        temporal = self.temporal(ports).transpose(1, 2).reshape(-1, 24, 4, 4)
        rf = self.aperture(temporal)
        condition = self.geometry(torch.cat((self.surface(image), summary[:, :32]), dim=1))
        gain, bias = self.modulation(condition).chunk(2, 1)
        rf = (1 + torch.tanh(gain)) * rf + bias
        return self.material(torch.cat((rf, self.summary(summary[:, 32:]), condition), dim=1))

    def forward(self, ports, image, summary):
        return self.target_mean + self.target_scale * self.normalized_output(ports, image, summary)


def _train_member(seed, x, y, v, truth, epochs):
    torch.manual_seed(seed)
    model = MaterialNetwork([g.shape[1:] for g in x])
    model.calibrate(x, y)
    opt = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.025)
    best, state, best_epoch = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(y)).split(56):
            pred = model(*(g[idx] for g in x))
            residual = (pred - y[idx]) / model.target_scale
            loss = nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.65)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4)
            opt.step()
        model.eval()
        with torch.no_grad():
            error = ((model(*v) - truth) / model.target_scale).abs().mean().item()
        if error < best:
            best, best_epoch = error, epoch
            state = {k: val.clone() for k, val in model.state_dict().items()}
        if epoch in (45, 85):
            for group in opt.param_groups:
                group["lr"] *= .4
        if epoch - best_epoch >= 35:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(*v).numpy()
    return state, prediction, dict(seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                                   validation_normalized_mae=best)


def baseline_predictions(baseline, archive, frames, split, expected_ids, camera):
    predictions = []
    with _archive(archive) as f:
        if baseline.settings != _settings(f) or baseline.meta["camera"] != camera:
            raise ValueError("Baseline acquisition or camera differs")
        ids = f["scene_ids"].asstr()
        for ref, i in zip(expected_ids, f["splits/" + split][:], strict=True):
            if ref != ids[i]:
                raise ValueError("Baseline scene IDs differ")
            depth = np.load(frames / ref / "depth.npy", allow_pickle=False)
            predictions.append(baseline.predict(wave=_signal(f, i, baseline.settings), depth=depth, camera=camera))
    return np.stack(predictions)


def fit(train, frames, baseline_path, output, epochs=125):
    camera = json.loads((frames / "camera.json").read_text())
    settings, ids, x, y, bg = read_split(train, frames, "train", camera)
    _, val_ids, v, truth, val_bg = read_split(train, frames, "validation", camera)
    if set(ids).intersection(val_ids) or not np.allclose(val_bg, bg):
        raise ValueError("Incompatible or overlapping training and validation sets")
    baseline = DepthMaterial(baseline_path)
    if baseline.meta["training_ids"] != ids or baseline.meta["validation_ids"] != val_ids:
        raise ValueError("The comparison baseline must use the same fitted scenes")
    base = baseline_predictions(baseline, train, frames, "validation", val_ids, camera)
    states, preds, history = [], [], []
    for seed in SEEDS:
        state, pred, summary = _train_member(seed, x, y, v, truth, epochs)
        states.append(state)
        preds.append(pred)
        history.append(summary)
        print(json.dumps(summary), flush=True)
    neural = np.mean(preds, axis=0)
    target = truth.numpy()
    blends = np.stack([w*neural + (1-w)*base for w in BLEND_WEIGHTS])
    weights = BLEND_WEIGHTS[np.abs(blends - target).mean(axis=1).argmin(axis=0)]
    val_maes = {k: np.abs(v-target).mean(0).tolist() for k,v in
                dict(kernel=base, neural=neural, blend=weights*neural+(1-weights)*base).items()}
    meta = dict(format_version=1, model_type="neural_depth_material", mode="fusion", settings=settings,
                training_ids=ids, validation_ids=val_ids, background=bg.tolist(), camera=camera,
                input_shapes=[list(g.shape[1:]) for g in x], members=history, baseline_sha256=hashlib.sha256(
                    baseline_path.read_bytes()).hexdigest(), neural_weights=weights.tolist(), validation_mae=val_maes,
                parameters_per_member=sum(p.numel() for p in MaterialNetwork([g.shape[1:] for g in x]).parameters()))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=meta, states=states), output / "ensemble.pt")
    return meta


class NeuralDepthMaterial:
    def __init__(self, path, baseline=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.meta = payload["meta"]
        if self.meta.get("model_type") != "neural_depth_material" or self.meta.get("format_version") != 1:
            raise ValueError("Unknown neural material checkpoint")
        self.settings = self.meta["settings"]
        self.models = []
        for state in payload["states"]:
            model = MaterialNetwork(self.meta["input_shapes"])
            model.load_state_dict(state)
            self.models.append(model.eval())
        if len(self.models) != len(self.meta["members"]) or not self.models:
            raise ValueError("Incomplete neural material ensemble")
        self.baseline = None
        if baseline is not None:
            if hashlib.sha256(Path(baseline).read_bytes()).hexdigest() != self.meta["baseline_sha256"]:
                raise ValueError("Blending requires the original baseline checkpoint")
            self.baseline = DepthMaterial(baseline)
        self.mode = "fusion"

    @torch.inference_mode()
    def predict(self, *, wave, depth, camera):
        if self.meta["camera"] != camera:
            raise ValueError("Depth camera differs from training")
        features = tuple(torch.from_numpy(x).unsqueeze(0) for x in sensor_features(wave, depth, camera, self.settings))
        preds = torch.stack([m(*features) for m in self.models]).mean(0)[0].numpy()
        if self.baseline is not None:
            weights = np.array(self.meta["neural_weights"])
            preds = weights*preds + (1-weights)*self.baseline.predict(wave=wave, depth=depth, camera=camera)
        return preds

    def predict_files(self, rf_scene, depth_file, camera_file):
        return self.predict(wave=calibrated_subset(Path(rf_scene), self.settings),
                            depth=np.load(depth_file, allow_pickle=False), camera=json.loads(Path(camera_file).read_text()))


def evaluate(data, frames, baseline_path, output):
    predictor = NeuralDepthMaterial(output / "ensemble.pt", baseline_path)
    meta = predictor.meta
    settings, ids, x, y, _ = read_split(data, frames, "test", meta["camera"])
    if set(ids).intersection(meta["training_ids"] + meta["validation_ids"]) or settings != meta["settings"]:
        raise ValueError("Overlapping or incompatible test data")
    if json.loads((frames / "camera.json").read_text()) != meta["camera"]:
        raise ValueError("Test camera differs from training")
    with torch.inference_mode():
        members = np.stack([m(*x).numpy() for m in predictor.models])
    neural = members.mean(0)
    base = baseline_predictions(predictor.baseline, data, frames, "test", ids, meta["camera"])
    w = np.asarray(meta["neural_weights"])
    predictions = dict(kernel=base, neural=neural, blend=w*neural + (1-w)*base)
    truth = y.numpy()
    errors = {k: np.abs(v - truth) for k,v in predictions.items()}
    draws = np.random.default_rng(9841).integers(len(ids), size=(10000,len(ids)))
    effects = {}
    for k in ("neural", "blend"):
        delta = errors[k] - errors["kernel"]
        effects[k+"_minus_kernel"] = dict(mean=delta.mean(0).tolist(),
            ci95=np.quantile(delta[draws].mean(1), [.025,.975], axis=0).T.tolist())
    summary = dict(properties=PARAMETERS, train=len(meta["training_ids"]), validation=len(meta["validation_ids"]),
        test=len(ids), parameters_per_member=meta["parameters_per_member"], seeds=SEEDS, members=meta["members"],
        neural_weights=w.tolist(), validation_mae=meta["validation_mae"],
        test_mae={k:e.mean(0).tolist() for k,e in errors.items()}, paired_mae_effects=effects,
        individual_member_test_mae=np.abs(members - truth).mean(1).tolist(),
        bootstrap=dict(unit="scene", draws=10000, seed=9841))
    rows = [dict(scene=s, truth=t.tolist(), **{k:v[i].tolist() for k,v in predictions.items()})
            for i,(s,t) in enumerate(zip(ids,truth))]
    for name, obj in (("summary", summary), ("scenes", rows)):
        (output / (name + ".json")).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=2)
    commands = p.add_subparsers(dest="command", required=True)
    study = commands.add_parser("study")
    for name in ("train", "test", "frames", "baseline", "output"):
        study.add_argument("--"+name, type=Path, required=True)
    infer = commands.add_parser("infer")
    for name in ("model", "rf-scene", "depth", "camera"):
        infer.add_argument("--"+name, type=Path, required=True)
    infer.add_argument("--baseline", type=Path, help="Apply the validation-selected component weights")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "study":
        fit(args.train,args.frames,args.baseline,args.output)
        report = evaluate(args.test,args.frames,args.baseline,args.output)
    else:
        predictor = NeuralDepthMaterial(args.model,args.baseline)
        report = dict(zip(PARAMETERS,predictor.predict_files(args.rf_scene,args.depth,args.camera).tolist()))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
