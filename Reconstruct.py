"""Reconstruct a laboratory scene from its measured and empty RF recordings.

Compact single mode estimates occupancy, distance and material. Compact mixed mode estimates geometry,
or fits one or two RF-selected inclusions and their materials with --refine.
Metre mode returns a one-object occupancy mask and geometry estimate on the native lattice.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from CalibrateReference import CLEARANCE_HELP, calibrate_scene
from EchoKernel import EchoKernel
from EchoMaterial import EchoMaterial


RELEASE = "phantom-benchmark-v1"
RELEASE_URL = f"https://github.com/ja99/RFBasedNERFsModels/releases/download/{RELEASE}"
MANIFEST = Path(__file__).resolve().parent / "pretrained" / "manifest.json"
DEFAULT_SIMULATOR = Path(__file__).resolve().parent.parent / "RFBasedNERFsSimulator"
# The two released 40³ acquisitions use distinct cell spacings. Model inference
# subsequently checks the complete acquisition, including the array and clock.
SPACING_M = {"dielectric": 0.0045, "conductive": 0.0011785188139428508}


def release_asset(name, weights_dir, manifest):
    """Return a verified cache entry, atomically installing or repairing it if needed."""
    if manifest.get("release") != RELEASE or Path(name).name != name:
        raise ValueError(f"Expected the {RELEASE} weights manifest and a release asset filename")
    entry = manifest.get("assets", {}).get(name, {})
    digest, size = entry.get("sha256"), entry.get("bytes")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError(f"Missing or invalid SHA256 for {name} in the release manifest")
    if type(size) is not int or size <= 0:
        raise ValueError(f"Missing or invalid size for {name} in the release manifest")
    cache = Path(weights_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if target.is_file() and target.stat().st_size == size:
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                return target

    with TemporaryDirectory(dir=cache) as scratch:
        local = Path(scratch) / name
        received, checksum = 0, hashlib.sha256()
        with urlopen(f"{RELEASE_URL}/{name}", timeout=60) as remote, local.open("wb") as dest:
            for block in iter(lambda: remote.read(1024 * 1024), b""):
                received += len(block)
                if received > size:
                    raise ValueError(f"Downloaded size exceeds the release manifest for {name}")
                checksum.update(block)
                dest.write(block)
        if received != size or checksum.hexdigest() != digest:
            raise ValueError(f"Downloaded size or SHA256 differs from the release manifest for {name}")
        local.replace(target)
    return target


def acquisition_protocol(acquisition):
    spacing = acquisition.get("grid_spacing_m")
    if isinstance(spacing, (int, float)):
        for protocol, value in SPACING_M.items():
            if math.isclose(spacing, value, rel_tol=1e-9):
                return protocol
    raise ValueError(f"No released compact acquisition uses grid_spacing_m={spacing!r}")


def reconstruct(scene, output, *, mode="single", refine=False, device="cpu", weights_dir=None, simulator=None,
                clearance_cells=None, scene_id=None, depth=None, camera=None, depth_material_model=None, geometry_model=None,
                depth_geometry=False, depth_tolerance_mm=0.0, material_blend_baseline=None, geometry_completion_model=None, material_uncertainty=False):
    """Predict on the protocol’s native grid using its paired receiver recordings."""
    if mode not in ("single", "mixed", "metre") or device not in ("cpu", "cuda"):
        raise ValueError("Choose single, mixed or metre mode and a cpu or cuda device")
    if scene_id is not None and mode != "metre":
        raise ValueError("--scene-id selects a recording from the metre archive")
    if refine and mode == "metre":
        raise ValueError("Waveform refinement requires compact single or mixed mode")
    sensor_inputs = (depth, camera, depth_material_model)
    use_depth = any(path is not None for path in sensor_inputs)
    if use_depth and (not all(path is not None for path in sensor_inputs) or mode != "single" or refine):
        raise ValueError("Depth fusion requires --depth, --camera and --depth-material-model in learned compact single mode")
    if material_blend_baseline is not None and (not use_depth or Path(depth_material_model).suffix != ".pt"):
        raise ValueError("--material-blend-baseline requires a neural .pt depth material model")
    if material_uncertainty and (not use_depth or material_blend_baseline is not None):
        raise ValueError("--material-uncertainty requires a calibrated beta material checkpoint without blending")
    if geometry_completion_model is not None:
        if geometry_model is not None:
            raise ValueError("Choose either --geometry-completion-model or --geometry-model")
        depth_geometry = True
    if depth_geometry and not use_depth:
        raise ValueError("--depth-geometry requires the calibrated depth-fusion inputs")
    if not np.isfinite(depth_tolerance_mm) or depth_tolerance_mm < 0 or (depth_tolerance_mm and not depth_geometry):
        raise ValueError("Choose a nonnegative finite --depth-tolerance-mm with --depth-geometry")
    if geometry_model is not None and not use_depth:
        raise ValueError("--geometry-model requires the depth-fusion path")
    scene, output = Path(scene).resolve(), Path(output).expanduser()
    protected = ("acquisition.json", "sensor_samples.npy", "empty_sensor_samples.npy", "reference_calibration.json",
                 "scene.json", "grid.npy")
    input_paths = {Path(p).expanduser().resolve() for p in (*sensor_inputs, geometry_model, material_blend_baseline, geometry_completion_model) if p is not None}
    if output.resolve() in input_paths | {scene / name for name in protected}:
        raise ValueError("Choose a separate reconstruction output path")
    manifest = json.loads(MANIFEST.read_text())
    cache = weights_dir if weights_dir is not None else Path.home() / ".cache" / "rf-thesis"
    if mode == "metre":
        from MeterReconstruction import predict_scene

        checkpoint = release_asset("metre-single-shot.npz", cache, manifest)
        predict_scene(scene, checkpoint, scene_id, output, clearance_cells=clearance_cells)
        return output
    acquisition = json.loads((scene / "acquisition.json").read_text())
    protocol = acquisition_protocol(acquisition)
    if mode == "mixed" and protocol != "dielectric":
        raise ValueError("The released mixed models require the compact dielectric acquisition")
    if refine:
        filename = "RefineInstances.py" if mode == "mixed" else "RefinePhantom.py"
        script = Path(simulator or DEFAULT_SIMULATOR).expanduser().resolve()
        if script.is_dir():
            script = script / filename
        if not script.is_file():
            raise FileNotFoundError(f"Pass --simulator pointing to the checkout containing {filename}: {script}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        prediction = Path(scratch) / "prediction.npz"
        calibration = None
        if clearance_cells is not None:
            corrected = Path(scratch) / "reference" / scene.name
            calibration = calibrate_scene(scene, corrected, clearance_cells)
            scene = corrected
        if mode == "single":
            sensor_info = {}
            if geometry_completion_model is None:
                kernel_path = geometry_model or release_asset(f"{protocol}-kernel.npz", cache, manifest)
                if output.resolve() == Path(kernel_path).resolve():
                    raise ValueError("Choose an output path separate from the geometry checkpoint")
                kernel = EchoKernel(kernel_path)
            if use_depth:
                from DepthMaterial import DepthMaterial, calibrated_subset

                frame = np.load(depth, allow_pickle=False)
                calibration_spec = json.loads(Path(camera).read_text())
                if geometry_completion_model is not None:
                    from DepthCompletion import Completion
                    from EchoKernel import occupancy_sdf

                    completion = Completion(geometry_completion_model, simulator=simulator)
                    uses_rf = completion.meta["mode"] == "fusion"
                    wave = calibrated_subset(scene, completion.settings) if uses_rf else None
                    mask = completion.predict(frame, calibration_spec, wave, depth_tolerance_mm / 1000.)
                    sdf = occupancy_sdf(mask)
                    geometry_settings = completion.settings if uses_rf else None
                    sensor_info["geometry_source"] = "rf_depth_completion" if uses_rf else "depth_completion"
                    sensor_info["depth_tolerance_mm"] = depth_tolerance_mm
                else:
                    geometry_settings = kernel.settings
                    sdf = kernel.predict(calibrated_subset(scene, kernel.settings))
                    sensor_info["geometry_source"] = "rf"
                if material_uncertainty:
                    from BetaMaterial import BetaMaterial

                    fused = BetaMaterial(depth_material_model)
                    sensor_info["material_estimator"] = fused.meta["loss"]
                elif Path(depth_material_model).suffix == ".pt":
                    from NeuralDepthMaterial import NeuralDepthMaterial

                    fused = NeuralDepthMaterial(depth_material_model, material_blend_baseline)
                    sensor_info["material_estimator"] = "neural_kernel_blend" if material_blend_baseline else "neural"
                else:
                    fused = DepthMaterial(depth_material_model)
                if fused.mode != "fusion" or fused.settings["grid_shape"] != list(sdf.shape):
                    raise ValueError("Expected a registered RF and depth fusion model for this reconstruction grid")
                estimate = fused.predict(wave=calibrated_subset(scene, fused.settings), depth=frame,
                                         camera=calibration_spec)
                if material_uncertainty:
                    estimate, variance = estimate
                    sensor_info["inclusion_material_mean"] = estimate
                    sensor_info["inclusion_material_variance"] = variance
                    sensor_info["inclusion_material_parameters"] = ["relative_permittivity", "conductivity_s_per_m"]
                if depth_geometry and geometry_completion_model is None:
                    from DepthGeometry import constrain, registered_camera
                    from EchoKernel import occupancy_sdf

                    optics = registered_camera(acquisition, sdf.shape, calibration_spec, simulator)
                    free, surface = optics.observations(frame, tolerance_m=depth_tolerance_mm / 1000.)
                    sdf = occupancy_sdf(constrain(sdf < 0, free, surface))
                    sensor_info["geometry_source"] = "rf_and_depth"
                    sensor_info["depth_tolerance_mm"] = depth_tolerance_mm
                materials = np.where((sdf < 0)[..., None], estimate, fused.meta["background"]).astype("f4")
                sensor_info["material_source"] = "rf_and_depth"
                for name, settings in (("geometry", geometry_settings), ("material", fused.settings)):
                    if settings is None:
                        sensor_info[name + "_beam_indices"] = np.array([], dtype=int)
                        sensor_info[name + "_receiver_indices"] = np.array([], dtype=int)
                        continue
                    acq = settings["acquisition"]
                    sensor_info[name + "_beam_indices"] = acq.get("acquired_beam_indices", list(range(settings["sensor_shape"][1])))
                    sensor_info[name + "_receiver_indices"] = acq.get("receive_channel_indices", list(range(settings["sensor_shape"][0])))
            else:
                sdf = kernel.predict_scene(scene)
            if refine:
                initial = Path(scratch) / "initial.npy"
                np.save(initial, sdf)
            else:
                if not use_depth:
                    model = EchoMaterial(release_asset(f"{protocol}-material.npz", cache, manifest))
                    materials = model.compose(model.predict_scene(scene), sdf)
                np.savez_compressed(prediction, sdf_cells=sdf, occupied=sdf < 0, permittivity=materials[..., 0],
                                    conductivity_s_per_m=materials[..., 1], grid_origin_cells=acquisition["grid_origin_cells"],
                                    grid_spacing_m=acquisition["grid_spacing_m"], scene_id=scene.name, **sensor_info)
        elif refine:
            from EchoSet import SetPredictor

            checkpoint = release_asset("mixed-set.pt", cache, manifest)
            _, masks, centers, probability = SetPredictor(checkpoint, device).predict_scene(scene)
            initial = Path(scratch) / "instances.npz"
            np.savez_compressed(initial, masks=masks, centers_cells=centers, probability_two=probability)
        else:
            from EchoReconstructor import predict

            checkpoint = release_asset("mixed-volume.pt", cache, manifest)
            args = SimpleNamespace(checkpoint=checkpoint, scene_dir=scene, scene_id=None,
                                   device=device, output=prediction)
            with redirect_stdout(io.StringIO()):
                predict(args)
        if refine:
            material = "thesis" if protocol == "conductive" else "dielectric"
            # The sibling projects have distinct modules with identical names.
            subprocess.run([sys.executable, str(script), str(scene), str(initial), str(prediction),
                            "--material", material, "--device", device], check=True)
        if calibration is not None:
            with ZipFile(prediction, "a", compression=ZIP_DEFLATED) as archive:
                with archive.open("reference_calibration_json.npy", "w") as record:
                    np.save(record, json.dumps(calibration, allow_nan=False), allow_pickle=False)
        prediction.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Compact raw scene directory or metre signals.npz recording archive")
    parser.add_argument("output", type=Path, help="Destination prediction NPZ")
    parser.add_argument("--mode", choices=("single", "mixed", "metre"), default="single",
                        help="Single compact material and geometry, mixed compact geometry, or metre single-shot geometry")
    parser.add_argument("--scene-id", help="Select a recording from a multi-scene metre archive")
    parser.add_argument("--refine", action="store_true",
                        help="Fit compact geometry and materials to the waveform, using one or two RF-selected inclusions in mixed mode")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Device for mixed mode or refinement")
    parser.add_argument("--clearance-cells", type=int,
                        help=CLEARANCE_HELP + ". Enables gain correction and stores its diagnostics in the output NPZ.")
    parser.add_argument("--depth", type=Path, help="Registered first-return depth frame in metres for learned single mode")
    parser.add_argument("--depth-geometry", action="store_true", help="Constrain RF geometry by visible depth and free space")
    parser.add_argument("--depth-tolerance-mm", type=float, default=0.,
                        help="Depth uncertainty half-width in mm for carving and surface pinning (ideal frames: 0)")
    parser.add_argument("--camera", type=Path, help="Depth camera calibration JSON")
    parser.add_argument("--depth-material-model", type=Path, help="Trained kernel (.npz) or neural (.pt) depth material model")
    parser.add_argument("--material-uncertainty", action="store_true",
                        help="Use BetaMaterial mean fields and save inclusion-level marginal predictive variances")
    parser.add_argument("--material-blend-baseline", type=Path, help="Kernel checkpoint for the validation-selected neural blend")
    parser.add_argument("--geometry-completion-model", type=Path,
                        help="Predict convex support behind observed depth surfaces using a DepthCompletion checkpoint")
    parser.add_argument("--geometry-model", type=Path, help="Depth-fusion geometry checkpoint override, e.g. a broadside-only EchoKernel")
    parser.add_argument("--weights-dir", type=Path, help="Verified weights cache (default ~/.cache/rf-thesis)")
    parser.add_argument("--simulator", type=Path, help="Simulator checkout for --refine or --depth-geometry (default sibling checkout)")
    args = parser.parse_args()
    output = reconstruct(**vars(args))
    print(json.dumps({"output": str(output), "mode": args.mode, "refined": args.refine}))


if __name__ == "__main__":
    main()
