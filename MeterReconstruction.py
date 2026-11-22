"""Reconstruct one dielectric phantom from a single broadside pulse and a matched empty recording.

Fit on a training/validation signals.npz, then evaluate once on a separate test archive.
Predict needs sensor_samples, empty_sensor_samples and acquisition_json, plus scene_ids for batches.
The extractor uses the known array coordinates, sample clock and signed measured-minus-empty voltages.
It fits an apparent scatter point to the first arrivals, corrects it from training echoes, and decodes
one centrally symmetric solid per nonzero echo. Model and radius selection use full-volume validation IoU.

    python MeterReconstruction.py train --data training --checkpoint model.npz
    python MeterReconstruction.py evaluate --data sealed --checkpoint model.npz --output score.json
    python MeterReconstruction.py predict --data raw.npz --checkpoint model.npz --output occupancy.npz
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from CalibrateReference import CLEARANCE_HELP, calibrate_pair

C = 299792458.
VERSION = 1
SHAPE_KEYS = ("centers_cells", "kind", "radius_cells", "sizes_cells", "euler_deg")
LEADING = np.array([.012, .035, .09, .20, .45])
QUANTILES = np.array([.003, .015, .06, .20, .50, .80])


def archive(path):
    path = Path(path)
    return path / "signals.npz" if path.is_dir() else path


@dataclass
class Recordings:
    echo: np.ndarray
    ids: np.ndarray
    split: np.ndarray
    acquisition: dict
    reference_calibration: dict | None = None


def recordings(path, *, single=False, scene_id=None, clearance_cells=None):
    # Geometry members of an NPZ are never opened by prediction.
    with np.load(archive(path), allow_pickle=False) as data:
        raw = data["sensor_samples"].astype(np.float64)
        empty = data["empty_sensor_samples"].astype(np.float64)
        acquisition = json.loads(data["acquisition_json"].item())
        if "scene_ids" in data.files:
            ids = data["scene_ids"].astype(str)
        elif len(raw) == 1:
            ids = np.array(["scene"])
        else:
            raise ValueError("Expected scene_ids for a batch of recordings")
        split = data["split"].astype(str) if "split" in data.files else np.full(len(ids), "")
    expected = (len(acquisition["antenna_positions_cells"]), 1, acquisition["samples_per_shot"])
    if raw.shape != (len(ids), *expected) or empty.shape != expected:
        raise ValueError(f"Expected [scenes,{expected[0]},1,{expected[2]}] voltages and a matched shared empty shot")
    if split.shape != ids.shape or ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected a unique scene ID and split entry for each recording")
    if not (np.isfinite(raw).all() and np.isfinite(empty).all()):
        raise ValueError("Nonfinite recorded voltages")
    if acquisition["beamforming_angles_deg"] != [[0., 0.]] or acquisition["voltage_units"] != "V":
        raise ValueError("Expected a single broadside excitation and voltages in V")
    if single:
        if scene_id is None and len(ids) != 1:
            raise ValueError("Select --scene-id when predicting from multiple recordings")
        selected = np.flatnonzero(ids == scene_id) if scene_id is not None else np.array([0])
        if len(selected) != 1:
            raise ValueError(f"Scene ID {scene_id!r} absent")
        raw, ids, split = raw[selected], ids[selected], split[selected]
    calibration = None
    if clearance_cells is not None:
        if len(raw) != 1:
            raise ValueError("Fit direct-path gain separately for each measured recording")
        empty, calibration = calibrate_pair(raw[0], empty, acquisition, clearance_cells)
    return Recordings((raw - empty)[:, :, 0], ids, split, acquisition, calibration)


def geometry(path, indices):
    with np.load(archive(path), allow_pickle=False) as data:
        arrays = {k: data[k][indices] for k in SHAPE_KEYS}
    kinds = arrays["kind"].astype(str)
    count = len(kinds)
    if not np.isin(kinds, ["sphere", "box"]).all():
        raise ValueError("Expected sphere or box targets")
    for key in ("centers_cells", "sizes_cells", "euler_deg"):
        if arrays[key].shape != (count, 3) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key}")
    if arrays["radius_cells"].shape != (count,) or not np.isfinite(arrays["radius_cells"]).all():
        raise ValueError("Invalid radius_cells")
    if (arrays["radius_cells"] <= 0).any() or (arrays["sizes_cells"][kinds == "box"] <= 0).any():
        raise ValueError("Expected positive phantom dimensions")
    arrays["kind"] = kinds
    return arrays


def roi(acquisition):
    origin = np.asarray(acquisition["grid_origin_cells"], dtype=int)
    end = np.asarray(acquisition["grid_shape"], dtype=int) - acquisition["pml_thickness"]
    if origin.shape != (3,) or end.shape != (3,) or np.any(end <= origin):
        raise ValueError("Invalid acquisition ROI")
    return origin, end


def crossings(values, levels):
    # Linear interpolation between adjacent temporal samples of each trace.
    above = values[:, :, None] >= levels[None, None, :]
    idx = above.argmax(axis=1)
    prev = np.maximum(idx - 1, 0)
    lo = np.take_along_axis(values, prev, axis=1)
    hi = np.take_along_axis(values, idx, axis=1)
    frac = np.divide(levels - lo, hi-lo, out=np.zeros_like(hi), where=hi > lo)
    return np.where(above.any(axis=1), prev + np.clip(frac, 0., 1.), values.shape[1] - 1)


class TimingGeometry:
    def __init__(self, acquisition):
        self.tx = np.asarray(acquisition["transmit_positions_cells"], dtype=float)
        base = np.asarray(acquisition["antenna_positions_cells"], dtype=float)
        if self.tx.shape != base.shape or self.tx.ndim != 2 or self.tx.shape[1] != 3:
            raise ValueError("Invalid port geometry")
        self.rx = (self.tx + base) / 2
        self.aperture = self.rx.mean(axis=0)
        self.slope = np.linalg.pinv((self.rx - self.aperture)[:, 1:])
        self.dx = float(acquisition["grid_spacing_m"])
        self.scale = C * acquisition["time_step_s"] * acquisition["sample_stride"] / self.dx
        self.lo, self.end = roi(acquisition)
        if self.scale <= 0 or np.linalg.matrix_rank(self.slope) < 2:
            raise ValueError("Expected a two-dimensional receiver aperture and positive sampling interval")

    def locate(self, times):
        path = times * self.scale
        transverse = -self.slope @ (path - path.mean())
        transverse /= max(1., np.linalg.norm(transverse) / .975)
        direction = np.r_[np.sqrt(1 - transverse @ transverse), transverse]
        initial = self.aperture + direction * path.mean() / 2
        initial = np.clip(initial, self.lo + .001, self.end - 1.001)

        def distances(x):
            r = np.linalg.norm(x - self.rx, axis=1)
            s = np.linalg.norm(x - self.tx, axis=1)
            return r + s.min() - path

        def jacobian(x):
            r = x - self.rx
            s = x - self.tx
            n = np.linalg.norm(s, axis=1)
            j = n.argmin()
            return r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9) + s[j] / max(n[j], 1e-9)

        point = least_squares(distances, initial, jac=jacobian, bounds=(self.lo, self.end-1),
                              loss="soft_l1", f_scale=1., max_nfev=24).x
        return point, np.sqrt(np.mean(distances(point)**2))

    def features(self, echo):
        absolute = np.abs(echo)
        peak = absolute.max(axis=1)
        peak_safe = np.maximum(peak, max(peak.max()*1e-10, 1e-40))
        normalized = echo / peak_safe[:, None]
        leading = crossings(np.abs(normalized), LEADING)
        energy = (normalized**2).cumsum(axis=1)
        quantile = crossings(energy / np.maximum(energy[:, -1:], 1e-20), QUANTILES)
        times = np.column_stack((leading, quantile))
        rays = [self.locate(t) for t in [leading[:, 0], leading[:, 2], leading[:, 3], quantile[:, 1]]]
        points = np.stack([p for p, _ in rays])
        apparent = np.median(points[:3], axis=0)
        log_peak = np.log(peak_safe)
        timing = np.concatenate([points.ravel(), np.asarray([e for _, e in rays]), times.mean(axis=0),
                                 (times-times.mean(axis=0)).ravel(), (times-leading[:, 2:3]).mean(axis=0),
                                 [log_peak.mean()], log_peak-log_peak.mean()])
        # Align each signed waveform independently to its interpolated leading edge. Differential
        # arrival times remain available in the timing group. Preserve inter-port amplitude, which
        # depends on angle and material, separately from the normalized waveforms.
        t = np.arange(echo.shape[1])
        offsets = np.arange(-5, min(85, echo.shape[1]-1), 1.5)
        aligned = np.asarray([np.interp(delay+offsets, t, signal, left=0., right=0.)
                              for signal, delay in zip(normalized, leading[:, 2])])
        signed_shape = np.r_[aligned.mean(axis=0), aligned.ravel()]
        norm = np.linalg.norm(echo)
        signature = (echo / max(float(norm), 1e-40)).astype(np.float32).ravel()
        return timing, signed_shape, apparent, signature


def features(data):
    sensor = TimingGeometry(data.acquisition)
    return tuple(np.stack(group) for group in zip(*(sensor.features(echo) for echo in data.echo)))


def quartic(unit):
    x, y, z = unit.T
    a, b, c = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    return np.stack((a*a, b*b, c*c, a*b, a*c, b*c, a*xy, a*xz, b*xy, b*yz,
                     c*xz, c*yz, a*yz, b*xz, c*xy), axis=1)


def shape_projection():
    n = 240
    z = (np.arange(n) + .5) * 2/n - 1
    angle = np.arange(n) * (np.pi*(3 - np.sqrt(5)))
    r = np.sqrt(1-z*z)
    dirs = np.column_stack((r*np.cos(angle), r*np.sin(angle), z))
    return dirs, np.linalg.pinv(quartic(dirs))


def shape_vectors(labels):
    dirs, project = shape_projection()
    centers, radius = labels["centers_cells"], labels["radius_cells"]
    out = np.zeros((len(radius), 19))
    out[:, :3] = centers
    out[:, 3] = np.log(radius)
    for i in np.flatnonzero(labels["kind"] == "box"):
        rot = Rotation.from_euler("xyz", labels["euler_deg"][i], degrees=True).as_matrix()
        radial = (labels["sizes_cells"][i]/2 / np.maximum(np.abs(dirs @ rot), 1e-15)).min(axis=1)
        out[i, 4:] = project @ (radial / radius[i] - 1.)
    return out


def box_points(center, extent, acquisition):
    origin, end = roi(acquisition)
    low = np.maximum(np.floor(center-extent).astype(int), origin)
    high = np.minimum(np.ceil(center+extent).astype(int) + 1, end)
    if np.any(low >= high):
        return np.empty((0, 3), dtype=int)
    return np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(low, high)), indexing="ij"), axis=-1).reshape(-1, 3)


def flat_indices(points, acquisition):
    origin, end = roi(acquisition)
    return np.ravel_multi_index((points - origin).T, end - origin)


def truth_voxels(labels, i, acquisition):
    center, radius = labels["centers_cells"][i], labels["radius_cells"][i]
    sphere = labels["kind"][i] == "sphere"
    if sphere:
        extent = np.full(3, radius)
    else:
        rot = Rotation.from_euler("xyz", labels["euler_deg"][i], degrees=True).as_matrix()
        half = labels["sizes_cells"][i]/2
        extent = np.abs(rot) @ half
    points = box_points(center, extent, acquisition)
    delta = points - center
    tolerance = 16*np.finfo(float).eps * max(1., np.abs([center-extent, center+extent]).max(), (2*extent).max())
    if sphere:
        inside = np.linalg.norm(delta, axis=1) < radius - tolerance
    else:
        inside = np.all(np.abs(delta @ rot) < half - tolerance, axis=1)
    return flat_indices(points[inside], acquisition)


def predicted_voxels(vector, acquisition):
    if np.isneginf(vector[3]):
        return np.empty(0, dtype=np.intp)
    origin, end = roi(acquisition)
    center = vector[:3]
    radius = np.exp(np.clip(vector[3], np.log(.5), np.log((end-origin).min()/3)))
    # The learned radial correction represents one centrally symmetric solid. The hard bound
    # makes this AABB evaluation exactly equivalent to checking all ROI cells for membership.
    points = box_points(center, 1.85*radius, acquisition)
    delta = points-center
    distance = np.linalg.norm(delta, axis=1)
    basis = quartic(delta/np.maximum(distance[:, None], 1e-15))
    radial = np.clip(1 + np.einsum("ij,j->i", basis, vector[4:]), .5, 1.85)
    return flat_indices(points[distance < radius*radial], acquisition)


def mean_iou(vectors, targets, acquisition):
    return np.mean([overlap(predicted_voxels(v, acquisition), target)[0] for v, target in zip(vectors, targets)])


def overlap(predicted, target):
    intersection = np.intersect1d(predicted, target, assume_unique=True).size
    union = len(predicted) + len(target) - intersection
    return intersection / union if union else 1., intersection, union


def normalize(train, others):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    # Constant and roundoff-level features supply zero information in the training population.
    scale[scale < 1e-8] = np.inf
    transform = lambda x: np.clip((x-mean)/scale, -8., 8.) / np.sqrt(x.shape[1])
    return mean, scale, transform(train), transform(others)


def combined(timing, wave, wave_weight):
    return np.column_stack((timing*np.sqrt(1-wave_weight), wave*np.sqrt(wave_weight)))


def kernel(x, train, bandwidth):
    d2 = cdist(x, train, "sqeuclidean")
    return x @ train.T + np.exp(-d2 / bandwidth)


def regress(matrix, y, alpha):
    matrix = matrix.copy()
    matrix.flat[::matrix.shape[0]+1] += alpha
    return cho_solve(cho_factor(matrix, lower=True, check_finite=False), y, check_finite=False)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def train(path, checkpoint):
    data = recordings(path)
    if set(data.split) != {"train", "validation"}:
        raise ValueError("Fit requires only train and validation partitions, each nonempty")
    tr, val = (np.flatnonzero(data.split == key) for key in ("train", "validation"))
    if len(tr) < 2:
        raise ValueError("Fit needs at least two training scenes")
    timing, wave, apparent, signature = features(data)
    train_labels, val_labels = geometry(path, tr), geometry(path, val)
    train_shapes = shape_vectors(train_labels)
    target = train_shapes.copy()
    target[:, :3] -= apparent[tr]
    offset = target.mean(axis=0)
    tm, ts, tt, vt = normalize(timing[tr], timing[val])
    wm, ws, tw, vw = normalize(wave[tr], wave[val])
    true_voxels = [truth_voxels(val_labels, i, data.acquisition) for i in range(len(val))]
    valid_echo = np.any(data.echo[val] != 0., axis=(1, 2))
    trials, best = [], None
    for weight in (.10, .45):
        x, xv = combined(tt, tw, weight), combined(vt, vw, weight)
        distances = cdist(x, x, "sqeuclidean")
        median = max(float(np.median(distances[np.triu_indices(len(tr), 1)])), 1e-8)
        for width in (.45, 1.75):
            bandwidth = median*width
            row = kernel(xv, x, bandwidth)
            gram = kernel(x, x, bandwidth)
            for alpha in (.001, .025):
                coef = regress(gram, target-offset, alpha)
                estimates = row @ coef + offset
                estimates[:, :3] += apparent[val]
                estimates[~valid_echo, 3] = -np.inf
                value = float(mean_iou(estimates, true_voxels, data.acquisition))
                trials.append(dict(wave_weight=weight, bandwidth=bandwidth, ridge=alpha, mean_iou=value))
                if best is None or value > best[0]:
                    best = (value, weight, bandwidth, alpha, x, coef, estimates)
    _, weight, bandwidth, alpha, x, coef, estimate = best
    scales = [1., .94, 1.06, 1.12]
    scores = []
    for size in scales:
        adjusted = estimate.copy()
        adjusted[:, 3] += np.log(size)
        scores.append(float(mean_iou(adjusted, true_voxels, data.acquisition)))
    picked = int(np.argmax(scores))
    val_report = dict(validation_scenes=len(val), mean_scene_iou=scores[picked], size_multiplier=scales[picked],
                      validation_trials=trials, validation_radius_trials=dict(zip(map(str, scales), scores)))
    atomic_npz(checkpoint, version=VERSION, acquisition_json=json.dumps(data.acquisition, sort_keys=True),
               timing_mean=tm, timing_scale=ts, wave_mean=wm, wave_scale=ws, wave_weight=weight,
               bandwidth=bandwidth, ridge=alpha, train_x=x, coef=coef, offset=offset,
               size_multiplier=scales[picked], train_signature=signature[tr], mean_shape=train_shapes.mean(axis=0),
               train_ids=data.ids[tr], validation_ids=data.ids[val], validation_json=json.dumps(val_report),
               **{f"train_{k}": v for k, v in train_labels.items()})
    return val_report


class Estimator:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as state:
            if state["version"].item() != VERSION:
                raise ValueError("Unknown metre reconstruction checkpoint")
            self.state = {k: state[k] for k in state.files}
        self.acquisition = json.loads(self.state["acquisition_json"].item())

    def predict(self, data, controls=False):
        if data.acquisition != self.acquisition:
            raise ValueError("Acquisition differs from the checkpoint")
        timing, wave, apparent, signature = features(data)
        s = self.state
        t = np.clip((timing - s["timing_mean"]) / s["timing_scale"], -8., 8.) / np.sqrt(timing.shape[1])
        w = np.clip((wave - s["wave_mean"]) / s["wave_scale"], -8., 8.) / np.sqrt(wave.shape[1])
        row = kernel(combined(t, w, s["wave_weight"]), s["train_x"], s["bandwidth"])
        predicted = row @ s["coef"] + s["offset"]
        predicted[:, :3] += apparent
        predicted[:, 3] += np.log(s["size_multiplier"])
        silent = ~np.any(data.echo != 0., axis=(1, 2))
        predicted[silent, :3] = np.nan
        predicted[silent, 3] = -np.inf
        if not controls:
            return predicted
        prior = np.broadcast_to(s["mean_shape"], predicted.shape).copy()
        timing_only = prior.copy()
        timing_only[:, :3] = apparent + s["offset"][:3]
        timing_only[silent] = predicted[silent]
        nearest = (signature @ s["train_signature"].T).argmax(axis=1)
        return predicted, timing_only, prior, nearest


def occupancy_centroid(indices, acquisition):
    if len(indices):
        origin, end = roi(acquisition)
        return np.mean(np.unravel_index(indices, end - origin), axis=1) + origin
    return None


def score(vectors, masks, true_masks, labels, ids, acquisition):
    cm = acquisition["grid_spacing_m"] * 100
    rows = []
    aperture = TimingGeometry(acquisition).aperture
    for i, (vec, pred, actual) in enumerate(zip(vectors, masks, true_masks)):
        value, intersection, union = overlap(pred, actual)
        a = occupancy_centroid(pred, acquisition)
        b = occupancy_centroid(actual, acquisition)
        center = labels["centers_cells"][i]
        diff = vec[:3] - center
        located = np.isfinite(diff).all()
        rows.append(dict(scene_id=str(ids[i]), kind=str(labels["kind"][i]), occupancy_iou=value,
                         occupied_intersection=intersection, occupied_union=union, occupied_predicted=len(pred),
                         occupied_target=len(actual), predicted_center_cells=vec[:3].tolist() if located else None,
                         target_slant_range_m=float(np.linalg.norm(center-aperture)*cm/100),
                         target_axial_distance_m=float((center-aperture)[0]*cm/100),
                         center_error_cm=float(np.linalg.norm(diff)*cm) if located else None,
                         axial_error_cm=float(diff[0]*cm) if located else None,
                         lateral_error_cm=float(np.linalg.norm(diff[1:])*cm) if located else None,
                         occupancy_centroid_error_cm=float(np.linalg.norm(a-b)*cm) if a is not None and b is not None else None))

    def summarize(group):
        located = [r for r in group if r["center_error_cm"] is not None]
        err = np.array([r["center_error_cm"] for r in located])

        ious = np.array([r["occupancy_iou"] for r in group])
        total_union = sum(r["occupied_union"] for r in group)
        centroid = [r["occupancy_centroid_error_cm"] for r in group if r["occupancy_centroid_error_cm"] is not None]
        return dict(scenes=len(group), mean_scene_iou=float(ious.mean()), median_scene_iou=float(np.median(ious)),
                    pooled_iou=sum(r["occupied_intersection"] for r in group)/total_union if total_union else 1.,
                    located_scenes=len(located), mean_center_error_cm=float(err.mean()) if len(err) else None,
                    median_center_error_cm=float(np.median(err)) if len(err) else None,
                    p90_center_error_cm=float(np.quantile(err, .9)) if len(err) else None,
                    rmse_center_cm=float(np.sqrt((err**2).mean())) if len(err) else None,
                    mean_occupancy_centroid_error_cm=float(np.mean(centroid)) if centroid else None,
                    mean_absolute_axial_error_cm=float(np.mean([abs(r["axial_error_cm"]) for r in located])) if located else None,
                    mean_lateral_error_cm=float(np.mean([r["lateral_error_cm"] for r in located])) if located else None,
                    scenes_iou_at_least_50=int((ious>=.50).sum()))
    range_groups = {}
    for lo, hi in ((0., .4), (.4, .6), (.6, .8), (.8, 1.), (1., np.inf)):
        group = [r for r in rows if lo <= r["target_slant_range_m"] < hi]
        if group:
            range_groups[f"[{lo:.1f},{hi:.1f})"] = summarize(group)
    return dict(summary=summarize(rows), by_slant_range_m=range_groups,
                by_kind={str(k): summarize([r for r in rows if r["kind"] == k]) for k in np.unique(labels["kind"])},
                scenes=rows)


def evaluate(path, checkpoint):
    data, model = recordings(path), Estimator(checkpoint)
    s = model.state
    if set(data.split) != {"test"} or np.intersect1d(data.ids, np.r_[s["train_ids"], s["validation_ids"]]).size:
        raise ValueError("Evaluation needs independent test IDs disjoint from train and validation")
    y, ray, prior, nearest = model.predict(data, controls=True)
    template_labels = {k: s[f"train_{k}"][nearest] for k in SHAPE_KEYS}
    template = shape_vectors(template_labels)
    methods = {k: (v, [predicted_voxels(a, data.acquisition) for a in v])
               for k, v in [("reconstruction", y), ("ray_with_training_bias", ray), ("training_mean", prior)]}
    methods["nearest_signed_echo"] = (template, [truth_voxels(template_labels, i, data.acquisition) for i in range(len(y))])
    if len(y) > 1:
        order = np.random.default_rng(317).permutation(len(y))
        perm = np.empty_like(order)
        perm[order] = np.roll(order, 1)
        methods["shuffled_echo"] = (y[perm], [methods["reconstruction"][1][i] for i in perm])
    # All predictions, their voxels, and controls are frozen before opening held-out geometry.
    truth = geometry(path, np.arange(len(data.ids)))
    masks = [truth_voxels(truth, i, data.acquisition) for i in range(len(data.ids))]
    origin, end = roi(data.acquisition)
    return dict(roi_origin_cells=origin.tolist(), roi_shape=(end-origin).tolist(),
                grid_spacing_m=data.acquisition["grid_spacing_m"], training_scenes=len(s["train_ids"]),
                validation=json.loads(s["validation_json"].item()), test_scenes=len(y),
                results={k: score(v, pred, masks, truth, data.ids, data.acquisition) for k, (v, pred) in methods.items()})


def predict_scene(path, checkpoint, scene_id, output, *, clearance_cells=None):
    if Path(output).resolve() in (archive(path).resolve(), Path(checkpoint).resolve()):
        raise ValueError("Choose an output path distinct from the recording and checkpoint")
    one = recordings(path, single=True, scene_id=scene_id, clearance_cells=clearance_cells)
    model = Estimator(checkpoint)
    vector = model.predict(one)[0]
    origin, end = roi(one.acquisition)
    mask = np.zeros(tuple(end-origin), dtype=bool)
    mask.flat[predicted_voxels(vector, one.acquisition)] = True
    provenance = ({"reference_calibration_json": json.dumps(one.reference_calibration, allow_nan=False)}
                  if one.reference_calibration is not None else {})
    atomic_npz(output, occupancy=mask, center_cells=vector[:3], radius_scale_cells=np.exp(vector[3]),
               nonzero_echo=bool(one.echo.any()),
               quartic_coefficients=vector[4:], roi_origin_cells=origin, scene_id=str(one.ids[0]),
               acquisition_json=json.dumps(one.acquisition, sort_keys=True), **provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for key in ("train", "evaluate", "predict"):
        q = commands.add_parser(key)
        q.add_argument("--data", type=Path, required=True)
        q.add_argument("--checkpoint", type=Path, required=True)
        if key != "train":
            q.add_argument("--output", type=Path, required=True)
        if key == "predict":
            q.add_argument("--clearance-cells", type=int, help=CLEARANCE_HELP)
            q.add_argument("--scene-id", help="Required when the archive contains multiple recordings")
    args = p.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.checkpoint), indent=2))
    elif args.command == "evaluate":
        report = evaluate(args.data, args.checkpoint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k: v["summary"] for k, v in report["results"].items()}, indent=2))
    else:
        predict_scene(args.data, args.checkpoint, args.scene_id, args.output, clearance_cells=args.clearance_cells)


if __name__ == "__main__":
    main()
