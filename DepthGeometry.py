"""Constrain a compact RF mask with the free space and visible surface from a registered depth frame."""

import argparse
import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np

from EchoKernel import EchoKernel, _archive, _signal, _target


DEFAULT_SIMULATOR = Path(__file__).resolve().parent.parent / "RFBasedNERFsSimulator"


def registered_camera(acquisition, shape, calibration, simulator=None):
    """Check the full registration against the RF scene and load only the NumPy camera code."""
    source = Path(simulator or DEFAULT_SIMULATOR)
    if source.is_dir():
        source = source / "DepthCamera.py"
    if source.name != "DepthCamera.py" or not source.is_file():
        raise FileNotFoundError(f"Expected the simulator's DepthCamera.py at {source}")
    spec = spec_from_file_location("rf_depth_camera", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    optics = calibration["intrinsics"]
    camera = module.DepthCamera.from_acquisition(acquisition, shape, width=optics["width"],
                                               height=optics["height"], hfov_deg=optics["hfov_deg"])
    if camera.calibration != calibration:
        raise ValueError("Depth calibration does not match the RF crop, pose or camera intrinsics")
    return camera


def constrain(mask, free, surface):
    if (any(a.dtype != bool for a in (mask, free, surface)) or mask.ndim != 3
            or mask.shape != free.shape or mask.shape != surface.shape):
        raise ValueError("Expected registered Boolean occupancy and observation masks")
    return (mask & ~free) | surface


def iou(predicted, truth):
    union = np.count_nonzero(truth | predicted)
    return np.count_nonzero(truth & predicted) / union if union else 1.


def evaluate(data, model_path, frames, output, simulator=None):
    frames, output = Path(frames), Path(output)
    kernel = EchoKernel(model_path)
    camera = registered_camera(kernel.settings["acquisition"], kernel.settings["grid_shape"],
                               json.loads((frames / "camera.json").read_text()), simulator)
    rng = np.random.default_rng(84529)
    rows = []
    with _archive(data) as archive:
        kernel.check_archive(archive)
        for i in archive["splits/test"][:]:
            scene = archive["scene_ids"].asstr()[i]
            rf = kernel.predict(_signal(archive, i, kernel.settings)) < 0
            image = np.load(frames / scene / "depth.npy", allow_pickle=False)
            free, surface = camera.observations(image)
            fused = constrain(rf, free, surface)
            noisy = image.copy()
            valid = image > 0
            noisy[valid] = np.maximum(0, image[valid] + rng.normal(0, .001, valid.sum()))
            free_noisy, surface_noisy = camera.observations(noisy)
            fused_noisy = constrain(rf, free_noisy, surface_noisy)
            conservative = constrain(rf, *camera.observations(image, tolerance_m=.003))
            conservative_noisy = constrain(rf, *camera.observations(noisy, tolerance_m=.003))
            truth = _target(archive, i) < 0
            if (free & truth).any() or (surface & ~truth).any():
                raise ValueError(f"Clean optical observations inconsistent with geometry at {scene}")
            rows.append(dict(scene=scene, rf=iou(rf, truth), depth_surface=iou(surface, truth),
                             fusion=iou(fused, truth), depth_surface_1mm=iou(surface_noisy, truth),
                             fusion_1mm=iou(fused_noisy, truth), fusion_tol3mm=iou(conservative, truth),
                             fusion_1mm_tol3mm=iou(conservative_noisy, truth), true_cells=int(truth.sum()),
                             surface_cells=int(surface.sum()), free_cells=int(free.sum())))
    if not rows:
        raise ValueError("No held-out scenes")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scene_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draws = np.random.default_rng(132415).integers(len(rows), size=(10000, len(rows)))

    def interval(scores):
        return dict(mean=float(scores.mean()), ci95=np.quantile(scores[draws].mean(1), [.025, .975]).tolist())

    scores = {k: np.array([row[k] for row in rows]) for k in
              ("rf", "depth_surface", "fusion", "depth_surface_1mm", "fusion_1mm", "fusion_tol3mm", "fusion_1mm_tol3mm")}
    result = dict(scenes=len(rows), methods={k: interval(v) for k, v in scores.items()}, contrasts={},
                  depth_noise=dict(sd_m=.001, seed=84529, valid_returns_only=True),
                  paired_bootstrap=dict(draws=10000, seed=132415, unit="scene"))
    for first, second in (("fusion", "rf"), ("fusion", "depth_surface"), ("fusion_1mm", "rf"),
                          ("fusion_tol3mm", "rf"), ("fusion_1mm_tol3mm", "rf")):
        result["contrasts"][f"{first}_minus_{second}"] = interval(scores[first] - scores[second])
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "model", "frames", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--simulator", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.model, args.frames, args.output, args.simulator), indent=2))
