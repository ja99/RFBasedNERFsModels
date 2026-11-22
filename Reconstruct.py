"""Reconstruct a compact benchmark scene from its measured and empty RF recordings."""

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

import numpy as np

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


def reconstruct(scene, output, *, mode="single", refine=False, device="cpu", weights_dir=None, simulator=None):
    """Predict a native-grid NPZ from acquisition.json and the two receiver recordings."""
    if mode not in ("single", "mixed") or device not in ("cpu", "cuda"):
        raise ValueError("Choose single or mixed mode and a cpu or cuda device")
    if refine and mode != "single":
        raise ValueError("Waveform refinement requires single mode")
    scene, output = Path(scene).resolve(), Path(output).expanduser()
    protected = ("acquisition.json", "sensor_samples.npy", "empty_sensor_samples.npy", "scene.json", "grid.npy")
    if output.resolve() in {scene / name for name in protected}:
        raise ValueError("Choose a separate reconstruction output path")
    acquisition = json.loads((scene / "acquisition.json").read_text())
    protocol = acquisition_protocol(acquisition)
    if mode == "mixed" and protocol != "dielectric":
        raise ValueError("The released mixed volume model requires the compact dielectric acquisition")
    if refine:
        script = Path(simulator or DEFAULT_SIMULATOR).expanduser().resolve()
        if script.is_dir():
            script = script / "RefinePhantom.py"
        if not script.is_file():
            raise FileNotFoundError(f"Pass --simulator pointing to the checkout containing RefinePhantom.py: {script}")

    manifest = json.loads(MANIFEST.read_text())
    cache = weights_dir if weights_dir is not None else Path.home() / ".cache" / "rf-thesis"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        prediction = Path(scratch) / "prediction.npz"
        if mode == "mixed":
            from EchoReconstructor import predict

            checkpoint = release_asset("mixed-volume.pt", cache, manifest)
            args = SimpleNamespace(checkpoint=checkpoint, scene_dir=scene, scene_id=None,
                                   device=device, output=prediction)
            with redirect_stdout(io.StringIO()):
                predict(args)
        else:
            kernel = EchoKernel(release_asset(f"{protocol}-kernel.npz", cache, manifest))
            sdf = kernel.predict_scene(scene)
            if refine:
                initial = Path(scratch) / "initial.npy"
                np.save(initial, sdf)
                material = "thesis" if protocol == "conductive" else "dielectric"
                # The sibling projects have distinct modules with identical names.
                subprocess.run([sys.executable, str(script), str(scene), str(initial), str(prediction),
                                "--material", material, "--device", device], check=True)
            else:
                model = EchoMaterial(release_asset(f"{protocol}-material.npz", cache, manifest))
                materials = model.compose(model.predict_scene(scene), sdf)
                np.savez_compressed(prediction, sdf_cells=sdf, occupied=sdf < 0, permittivity=materials[..., 0],
                                    conductivity_s_per_m=materials[..., 1], grid_origin_cells=acquisition["grid_origin_cells"],
                                    grid_spacing_m=acquisition["grid_spacing_m"], scene_id=scene.name)
        prediction.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Directory containing acquisition.json and the two raw RF recordings")
    parser.add_argument("output", type=Path, help="Destination prediction NPZ")
    parser.add_argument("--mode", choices=("single", "mixed"), default="single")
    parser.add_argument("--refine", action="store_true", help="Fit one homogeneous inclusion to the measured waveform")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Device for mixed mode or refinement")
    parser.add_argument("--weights-dir", type=Path, help="Verified weights cache (default ~/.cache/rf-thesis)")
    parser.add_argument("--simulator", type=Path, help="Simulator checkout for --refine (default sibling checkout)")
    args = parser.parse_args()
    output = reconstruct(**vars(args))
    print(json.dumps({"output": str(output), "mode": args.mode, "refined": args.refine}))


if __name__ == "__main__":
    main()
