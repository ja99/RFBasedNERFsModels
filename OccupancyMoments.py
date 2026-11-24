"""Convert marginal support forecasts into occupancy moments using whole-shape calibration residuals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from BetaDepthCompletion import BetaCompletion
from DepthCompletion import decoded
from DepthMaterial import calibrated_subset
from EchoKernel import _archive, _settings, _signal, _support, _target


def fit_residuals(checkpoint, selection, data, frames, output):
    model = BetaCompletion(checkpoint)
    ids = json.loads(selection.read_text())["calibration_ids"]
    camera = json.loads((frames / "camera.json").read_text())
    residuals, standardized = [], []
    with _archive(data) as archive:
        if _settings(archive) != model.meta["settings"]:
            raise ValueError("Residual archive RF calibration differs from the checkpoint")
        val = {archive["scene_ids"].asstr()[i]: i for i in archive["splits/validation"][:]}
        if not set(ids) <= val.keys() or len(set(ids)) != len(ids):
            raise ValueError("Expected unique calibration scenes from the validation archive")
        for scene in ids:
            i = val[scene]
            z = np.load(frames / scene / "depth.npy")
            pred = model.predict(_signal(archive, i, model.meta["settings"]), z, camera)
            error = _support(_target(archive, i) < 0) - pred["expected_support_cells"]
            residuals.append(error)
            standardized.append(error / np.sqrt(pred["variance_support_cells2"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, residuals=residuals, standardized=standardized, ids=ids,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())


class OccupancyMoments:
    def __init__(self, checkpoint, bank):
        self.model = BetaCompletion(checkpoint)
        with np.load(bank, allow_pickle=False) as data:
            if data["checkpoint_sha256"].item() != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError("Residual calibration and geometry checkpoint differ")
            raw, scaled = data["residuals"], data["standardized"]
            if (raw.ndim != 2 or raw.shape != scaled.shape or len(raw) < 2 or
                    not np.isfinite(raw).all() or not np.isfinite(scaled).all()):
                raise ValueError("Expected finite whole-shape calibration residuals")
            self.standardized = scaled - scaled.mean(0)
            self.residuals = raw - raw.mean(0)

    def probability(self, forecast, depth, constant=False):
        free, surface = self.model.optics.observations(depth)
        delta = self.residuals if constant else self.standardized * np.sqrt(forecast["variance_support_cells2"])
        extension = forecast["expected_support_cells"] - forecast["visible_support_cells"]
        count = np.zeros_like(free, dtype=np.int32)
        for error in delta:
            count += decoded(forecast["visible_support_cells"], extension + error, self.model.meta["pad"], free, surface)
        return (count / len(delta)).astype(np.float32)

    def predict(self, wave, depth, camera):
        forecast = self.model.predict(wave, depth, camera)
        p = self.probability(forecast, depth)
        return dict(**forecast, occupancy_probability=p, occupancy_variance=p*(1-p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, args in [("fit", ("checkpoint","selection","data","frames","output")),
                       ("infer", ("checkpoint","bank","scene","depth","camera","output"))]:
        cmd = sub.add_parser(name)
        for arg in args:
            cmd.add_argument(arg, type=Path)
    a = parser.parse_args()
    torch.set_num_threads(1)
    if a.action == "fit":
        fit_residuals(a.checkpoint,a.selection,a.data,a.frames,a.output)
    else:
        model = OccupancyMoments(a.checkpoint, a.bank)
        wave = calibrated_subset(a.scene, model.model.meta["settings"])
        result = model.predict(wave, np.load(a.depth), json.loads(a.camera.read_text()))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.output, **result)
