"""Pair RF recordings with their scenes and prepare a scene-split training archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np


def split_scene_indices(count, seed=42):
    """Hold out distinct validation and test scenes whenever there are at least three."""
    indices = np.random.default_rng(seed).permutation(count)
    n_val = max(1, count // 10) if count >= 2 else 0
    n_test = max(1, count // 10) if count >= 3 else 0
    n_train = count - n_val - n_test
    return dict(train=indices[:n_train], validation=indices[n_train:n_train + n_val],
                test=indices[n_train + n_val:])


def _scene_dirs(root):
    names = ("grid.npy", "sensor_samples.npy", "empty_sensor_samples.npy", "acquisition.json")
    scenes = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
                    and any((d / name).exists() for name in names))
    if not scenes:
        raise ValueError(f"No scene directories found in {root}")
    for scene in scenes:
        for name in ("grid.npy", "sensor_samples.npy"):
            if not (scene / name).is_file():
                raise ValueError(f"Incomplete scene {scene.name}: missing {name}")
    return scenes


def _grid(scene, crop_x_end):
    values = np.load(scene / "grid.npy", mmap_mode="r", allow_pickle=False)
    if values.ndim != 4 or values.shape[-1] != 3 or any(n == 0 for n in values.shape):
        raise ValueError(f"{scene}: expected a nonempty [x,y,z,3] grid, found {values.shape}")
    if not 0 <= crop_x_end < values.shape[0]:
        raise ValueError(f"{scene}: invalid x crop {crop_x_end} for shape {values.shape}")
    return values[:values.shape[0] - crop_x_end]


def _material_digest(scene):
    # Match physical scenes even if acquired under different directory names or array dtypes.
    grid = _grid(scene, 0)
    digest = hashlib.sha256(str(grid.shape[:3]).encode())
    for plane in grid:
        material = np.ascontiguousarray(plane[..., :2], dtype="<f8")
        digest.update(memoryview(material).cast("B"))
    return digest.hexdigest()


def _load_trace(path, beamforming, layout=None):
    values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError(f"{path}: expected a finite, nonempty recording")
    if not beamforming:
        if values.ndim != 2:
            raise ValueError(f"{path}: expected [antenna,timestep]")
        return values[:, None, :]
    if values.ndim == 3:
        if layout == "direction,antenna,timestep":
            return values.transpose(1, 0, 2)
        if layout in (None, "antenna,direction,timestep"):
            if layout is None and values.shape[:2] == (5, 16):
                raise ValueError(f"{path}: declare direction-first recordings with --sensor-layout")
            return values
        raise ValueError(f"{path}: unsupported sensor layout {layout!r}")
    # An unidentified [16,750] file could equally be a single shot or five shorter shots.
    if values.ndim == 2 and layout == "packed-antenna-direction":
        if values.shape[0] == 80 or (values.shape[0] == 16 and values.shape[1] % 5 == 0):
            return values.reshape(16, 5, -1)
    if values.ndim == 2 and layout is None:
        raise ValueError(f"{path}: ambiguous 2-D recording. Use --single-shot or explicitly declare "
                         "--sensor-layout packed-antenna-direction for five antenna-major shots.")
    raise ValueError(f"{path}: expected named antenna/direction/time axes or a legacy packed recording")


def _training_scales(scenes, training, crop_x_end, signal, raw):
    grid_min, grid_max = np.full(3, np.inf), np.full(3, -np.inf)
    signal_min, signal_max = np.inf, -np.inf
    for i in training:
        grid = _grid(scenes[i], crop_x_end)
        grid_min = np.minimum(grid_min, grid.min(axis=(0, 1, 2)))
        grid_max = np.maximum(grid_max, grid.max(axis=(0, 1, 2)))
        wave = signal(scenes[i])
        signal_min = min(signal_min, wave.min())
        signal_max = max(signal_max, wave.max())
    if not (np.isfinite(grid_min).all() and np.isfinite(grid_max).all()):
        raise ValueError("Training grids need finite supervised targets. Use empty scenes as RF references.")
    # A constant training channel maps to zero and still has an invertible transform.
    grid_scale = np.where(grid_max > grid_min, grid_max - grid_min, 1.)
    scale = (signal_max - signal_min if raw else max(abs(signal_max), abs(signal_min))) or 1.
    return dict(grid_min=grid_min.tolist(), grid_max=grid_max.tolist(), grid_scale=grid_scale.tolist(),
                sensor_offset=float(signal_min) if raw else 0., sensor_scale=float(scale))


def _load_protocol(source):
    source = Path(source)
    with h5py.File(source / "dataset.hdf5" if source.is_dir() else source, "r") as archive:
        if "training_protocol" not in archive:
            raise ValueError(f"{source}: no saved training transforms or split in this archive")
        return json.loads(archive["training_protocol"].asstr()[()])


def _convert(data_path, *, beamforming, raw, reference_path, crop_x_end, seed, sensor_layout,
             normalization_from=None):
    root = Path(data_path)
    scenes = _scene_dirs(root)
    if raw and reference_path is not None:
        raise ValueError("Choose raw normalization or an empty reference, not both")

    metadata = []
    for scene in scenes:
        path = scene / "acquisition.json"
        meta = json.loads(path.read_text()) if path.is_file() else {}
        if not isinstance(meta, dict):
            raise ValueError(f"{path}: expected acquisition settings")
        if sensor_layout and meta.get("sensor_layout", sensor_layout) != sensor_layout:
            raise ValueError(f"{path}: declared sensor layout disagrees with requested layout")
        metadata.append(meta)
    # Acquisition metadata from the simulator describes the fixed sensor, grid and pulse.
    if any(meta != metadata[0] for meta in metadata[1:]):
        raise ValueError("Scenes have different acquisition settings. Check axes, beam angles and grid geometry.")
    meta = metadata[0]
    layout = meta.get("sensor_layout", sensor_layout)
    if not raw:
        for scene in scenes:
            reference = Path(reference_path) if reference_path is not None else scene / "empty_sensor_samples.npy"
            if not reference.is_file():
                raise ValueError(f"{scene}: missing matched empty_sensor_samples.npy. Supply --empty-reference "
                                 "from the same acquisition or select --raw for an uncalibrated archive.")

    shared_ref = None
    if reference_path is not None:
        reference_path = Path(reference_path)
        ref_metadata = reference_path.parent / "acquisition.json"
        ref_meta = json.loads(ref_metadata.read_text()) if ref_metadata.is_file() else {}
        if ref_meta != meta:
            raise ValueError(f"{reference_path}: shared reference needs matching acquisition.json metadata")
        shared_ref = _load_trace(reference_path, beamforming, layout)

    def signal(scene):
        measured = _load_trace(scene / "sensor_samples.npy", beamforming, layout)
        if raw:
            return measured
        empty = shared_ref
        if empty is None:
            empty = _load_trace(scene / "empty_sensor_samples.npy", beamforming, layout)
        if measured.shape != empty.shape:
            raise ValueError(f"{scene}: recording {measured.shape} and reference {empty.shape} differ")
        residual = measured - empty
        if not np.isfinite(residual).all():
            raise ValueError(f"{scene}: nonfinite empty-subtracted recording")
        return residual

    grid_shape = _grid(scenes[0], crop_x_end).shape
    sensor_shape = signal(scenes[0]).shape
    if "beamforming_angles_deg" in meta and len(meta["beamforming_angles_deg"]) != sensor_shape[1]:
        raise ValueError("Number of beam angles disagrees with the direction axis")
    if "samples_per_shot" in meta and meta["samples_per_shot"] != sensor_shape[2]:
        raise ValueError("Samples per shot disagree with the timestep axis")
    if "antenna_positions_cells" in meta and len(meta["antenna_positions_cells"]) != sensor_shape[0]:
        raise ValueError("Antenna positions disagree with the recording axis")
    if all(key in meta for key in ("grid_shape", "grid_origin_cells", "pml_thickness")):
        saved = np.array(meta["grid_shape"]) - np.array(meta["grid_origin_cells"]) - meta["pml_thickness"]
        if tuple(saved - [crop_x_end, 0, 0]) != grid_shape[:3]:
            raise ValueError("Grid shape disagrees with acquisition geometry and crop")

    sensor_storage = "antenna_direction,time" if beamforming else "time,antenna"
    settings = dict(grid_shape=list(grid_shape), sensor_shape=list(sensor_shape), sensor_storage=sensor_storage,
                    waveform_mode="legacy_raw" if raw else "calibrated", crop_x_end=crop_x_end,
                    coordinate_divisor=max(1, max(grid_shape[:3]) - 1), acquisition=meta)
    material_hashes = {scene.name: _material_digest(scene) for scene in scenes}
    if normalization_from is None:
        grouped = {}
        for i, digest in enumerate(material_hashes.values()):
            grouped.setdefault(digest, []).append(i)
        groups = list(grouped.values())
        splits = {key: np.array([i for g in group_ids for i in groups[g]], dtype=int)
                  for key, group_ids in split_scene_indices(len(groups), seed).items()}
        scales = _training_scales(scenes, splits["train"], crop_x_end, signal, raw)
        membership = {key: [scenes[i].name for i in indices] for key, indices in splits.items()}
        protocol = dict(settings=settings, normalization=scales, scene_ids=membership, material_hashes=material_hashes)
    else:
        protocol = _load_protocol(normalization_from)
        if settings != protocol["settings"]:
            raise ValueError("Evaluation and training inputs must have matching acquisition, grid and waveform settings")
        fitted_ids = set(protocol["scene_ids"]["train"] + protocol["scene_ids"]["validation"])
        fitted_hashes = {protocol["material_hashes"][name] for name in fitted_ids}
        overlap = sorted(name for name, digest in material_hashes.items() if name in fitted_ids or digest in fitted_hashes)
        if overlap:
            raise ValueError(f"Evaluation scenes overlap the fitted train/validation IDs or material grids: {overlap}")
        scales = protocol["normalization"]
        splits = dict(train=np.empty(0, dtype=int), validation=np.empty(0, dtype=int), test=np.arange(len(scenes)))
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    protocol_id = hashlib.sha256(protocol_json.encode()).hexdigest()
    grid_min, grid_scale = np.array(scales["grid_min"]), np.array(scales["grid_scale"])
    sdf_zero = np.float32(-grid_min[2] / grid_scale[2])

    output = root / "dataset.hdf5"
    with NamedTemporaryFile(dir=root, prefix=".dataset-", suffix=".hdf5", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(temporary, "w") as archive:
            archive.attrs.update(format_version=2, **scales, sensor_shape=sensor_shape,
                                 waveform_mode=settings["waveform_mode"], sensor_storage=sensor_storage,
                                 subsample_stride=5, crop_x_end=crop_x_end, preprocessing_id=protocol_id)
            if normalization_from is None:
                archive.attrs["split_seed"] = seed
            archive.attrs["acquisition_json"] = json.dumps(meta)
            archive.create_dataset("training_protocol", data=protocol_json, dtype=h5py.string_dtype())
            archive.create_dataset("scene_ids", data=[s.name for s in scenes], dtype=h5py.string_dtype())
            group = archive.create_group("splits")
            for key, indices in splits.items():
                group.create_dataset(key, data=indices)
            spatial_shape = grid_shape[:3]
            coords = np.indices(spatial_shape, dtype=np.float32).reshape(3, -1).T
            archive.create_dataset("positions", data=coords / settings["coordinate_divisor"])
            count = len(scenes)
            grids = archive.create_dataset("grids", (count, *grid_shape), dtype="f4")
            small_shape = tuple((n + 4) // 5 for n in spatial_shape)
            small = archive.create_dataset("subsampled_grids", (count, *small_shape, 3), dtype="f4")
            antennas, directions, steps = sensor_shape
            stored_shape = (antennas * directions, steps) if beamforming else (steps, antennas)
            signals = archive.create_dataset("sensor_samples", (count, *stored_shape), dtype="f4")
            for i, scene in enumerate(scenes):
                grid = _grid(scene, crop_x_end)
                wave = signal(scene)
                if grid.shape != grid_shape or wave.shape != sensor_shape:
                    raise ValueError(f"{scene}: inconsistent grid {grid.shape} or trace {wave.shape}")
                if not np.isfinite(grid).all():
                    raise ValueError(f"{scene}: nonfinite supervised grid")
                normalized = ((grid - grid_min) / grid_scale).astype(np.float32)
                # Preserve the occupied half-space when float32 rounding lands exactly on its boundary.
                sdf = normalized[..., 2]
                on_boundary = sdf == sdf_zero
                sdf[on_boundary & (grid[..., 2] < 0)] = np.nextafter(sdf_zero, np.float32(-np.inf))
                grids[i] = normalized
                small[i] = normalized[::5, ::5, ::5]
                wave = ((wave - scales["sensor_offset"]) / scales["sensor_scale"]).astype(np.float32)
                if not np.isfinite(normalized).all() or not np.isfinite(wave).all():
                    raise ValueError(f"{scene}: scaled values exceed float32 range")
                signals[i] = wave.reshape(stored_shape) if beamforming else wave[:, 0, :].T
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def save_to_hdf5_beamforming(data_path, *, raw=False, reference_path=None, crop_x_end=0, seed=42,
                             sensor_layout=None, normalization_from=None):
    return _convert(data_path, beamforming=True, raw=raw, reference_path=reference_path,
                    crop_x_end=crop_x_end, seed=seed, sensor_layout=sensor_layout, normalization_from=normalization_from)


def save_to_hdf5(data_path, *, raw=True, reference_path=None, crop_x_end=0, seed=42, normalization_from=None):
    """Convert historical single-shot recordings, retaining their [time,antenna] storage order."""
    return _convert(data_path, beamforming=False, raw=raw, reference_path=reference_path,
                    crop_x_end=crop_x_end, seed=seed, sensor_layout=None, normalization_from=normalization_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--single-shot", action="store_true", help="input files have [antenna,time] axes")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true", help="use historical raw min-max processing")
    mode.add_argument("--empty-reference", type=Path, help="one matched empty recording shared by all scenes")
    parser.add_argument("--sensor-layout", choices=["antenna,direction,timestep", "direction,antenna,timestep",
                                                   "packed-antenna-direction"])
    parser.add_argument("--crop-x-end", type=int, default=0, help="trim trailing x planes only for legacy data")
    parser.add_argument("--seed", type=int, default=42, help="seed for the saved scene split")
    parser.add_argument("--normalization-from", type=Path, help="prepare new test scenes using a fitted archive")
    args = parser.parse_args()
    print(_convert(args.data_path, beamforming=not args.single_shot, raw=args.raw,
                   reference_path=args.empty_reference, crop_x_end=args.crop_x_end, seed=args.seed,
                   sensor_layout=args.sensor_layout, normalization_from=args.normalization_from))
