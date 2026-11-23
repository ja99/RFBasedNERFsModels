"""Calibrated echo inversion for compact phantoms using RF features and convex support functions."""

import argparse
from itertools import product
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.signal import hilbert


def _directions():
    # Include all primitive lattice directions through a 5x5x5 neighborhood.
    axes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    directions = axes + [v for v in product(range(-2, 3), repeat=3)
                         if np.gcd.reduce(np.abs(v)) == 1 and v not in axes]
    values = np.asarray(directions, dtype='f4')
    return values / np.linalg.norm(values, axis=1, keepdims=True)


DIRECTIONS = _directions()
REPRESENTATIONS = ('voltage', 'iq', 'magnitude', 'voltage-envelope-stats')


def _archive(path):
    path = Path(path)
    return h5py.File(path / 'dataset.hdf5' if path.is_dir() else path, 'r')


def _settings(f):
    if f.attrs.get('waveform_mode') != 'calibrated' or not f.attrs.get('preprocessing_id'):
        raise ValueError('A calibrated archive with a saved training protocol is required')
    return dict(grid_shape=list(map(int, f['grids'].shape[1:-1])),
                sensor_shape=list(map(int, f.attrs['sensor_shape'])),
                preprocessing_id=str(f.attrs['preprocessing_id']),
                acquisition=json.loads(f.attrs['acquisition_json']),
                sensor_scale=float(f.attrs['sensor_scale']), sensor_offset=float(f.attrs['sensor_offset']))


def _signal(f, index, settings):
    values = f['sensor_samples'][index]
    if f.attrs['sensor_storage'] == 'time,antenna':
        values = values.T
    return values.reshape(settings['sensor_shape'])


def _target(f, index):
    return f['grids'][index, ..., 2] * f.attrs['grid_scale'][2] + f.attrs['grid_min'][2]


def _baseband(wave, acquisition):
    """Full-band analytic envelope with one common phase origin for every receiver and shot."""
    try:
        carrier = float(acquisition['frequency_hz'])
        step = float(acquisition['time_step_s'])
        stride = acquisition['sample_stride']
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('I/Q requires acquisition frequency_hz, time_step_s and sample_stride') from error
    if (not isinstance(stride, (int, np.integer)) or isinstance(stride, bool) or stride < 1 or
            not np.isfinite([carrier, step]).all() or min(carrier, step) <= 0 or carrier * step * stride >= .5):
        raise ValueError('I/Q requires a positive carrier below the sampled Nyquist frequency')
    steps = wave.shape[-1]
    # Append two windows of zeros before the periodic FFT Hilbert transform.
    # This is shift-equivalent to padding one window on either side.
    analytic = hilbert(wave.astype('f8'), N=3*steps, axis=-1)[..., :steps]
    oscillator = np.exp(-2j * np.pi * carrier * step * stride * np.arange(steps))
    return analytic * oscillator


def _echo_features(wave, representation='voltage', acquisition=None):
    wave = np.asarray(wave, dtype='f4')
    if wave.ndim != 3 or not np.isfinite(wave).all():
        raise ValueError('Expected a finite antenna-by-beam-by-time waveform')
    steps = wave.shape[-1]
    edges = np.linspace(0, steps, min(48, steps) + 1, dtype=int)
    if representation == 'voltage':
        signal = wave
        power = wave.astype('f8') ** 2
    elif representation in REPRESENTATIONS:
        baseband = _baseband(wave, acquisition)
        envelope = np.abs(baseband)
        power = envelope ** 2
        signal = {'iq': baseband, 'magnitude': envelope, 'voltage-envelope-stats': wave}[representation]
    else:
        raise ValueError(f'Unknown echo representation {representation!r}')
    pooled = np.add.reduceat(signal, edges[:-1], axis=-1) / np.diff(edges)
    if representation == 'iq':
        pooled = np.concatenate((pooled.real, pooled.imag), axis=-1)
    total = power.sum(axis=-1)
    rms = np.sqrt(total / steps)
    floor = max(rms.max() * 1e-6, 1e-12)
    signed = pooled / max(np.linalg.norm(pooled), 1e-12)
    waveform_rms, waveform_floor = rms, floor
    if representation == 'voltage-envelope-stats':
        # Keep both pooled-voltage groups byte-identical to the historical representation.
        waveform_rms = np.sqrt(np.sum(wave.astype('f8') ** 2, axis=-1) / steps)
        waveform_floor = max(waveform_rms.max() * 1e-6, 1e-12)
    unit = pooled / np.maximum(waveform_rms, waveform_floor)[..., None]
    cumulative = np.cumsum(power, axis=-1)
    fractions = [np.argmax(cumulative >= p * total[..., None], axis=-1) / max(steps - 1, 1)
                 for p in (0.01, 0.05, 0.15, 0.35, 0.5, 0.75, 0.95)]
    log_rms = np.log(np.maximum(rms, floor))
    # Relative response across antenna positions helps distinguish lateral
    # displacement from differences in the material's reflection coefficient.
    stats = np.stack([log_rms, log_rms - log_rms.mean(axis=0), *fractions])
    return tuple(a.ravel().astype('f4') for a in (signed, unit, stats))


def _reflections(settings):
    """Only mirror when a planar aperture and paired beam directions support it."""
    a, d, unused = settings['sensor_shape']
    base = np.asarray(settings['acquisition'].get('antenna_positions_cells', []))
    tips = np.asarray(settings['acquisition'].get('transmit_positions_cells', []))
    angles = np.asarray(settings['acquisition'].get('beamforming_angles_deg', []))
    origin = np.asarray(settings['acquisition'].get('grid_origin_cells', []))
    identity = (np.arange(a), np.arange(d), np.arange(len(DIRECTIONS)), np.zeros(len(DIRECTIONS)))
    variants = [identity]
    if (base.shape != (a, 3) or tips.shape != base.shape or angles.shape != (d, 2)
            or origin.shape != (3,) or np.ptp(base[:, 0]) != 0
            or not np.allclose(tips - base, tips[0] - base[0])):
        return variants
    aperture_center = (base.max(0) + base.min(0) + tips.max(0) + tips.min(0)) / 4 - origin
    polarization = np.asarray(settings['acquisition'].get('element_polarization_local', tips[0] - base[0]))
    for flip_y, flip_z in ((False, True), (True, False), (True, True)):
        flips = np.array((False, flip_y, flip_z))
        reflected_polarization = polarization * np.where(flips, -1, 1)
        if not (np.allclose(reflected_polarization, polarization)
                or np.allclose(reflected_polarization, -polarization)):
            continue
        positions = base.copy()
        positions[:, flips] = base[:, flips].min(0) + base[:, flips].max(0) - base[:, flips]
        reflected_angles = angles * np.where(flips[1:], -1, 1)
        ai = np.isclose(positions[:, None], base[None]).all(-1)
        di = np.isclose(reflected_angles[:, None], angles[None]).all(-1)
        if not ((ai.sum(1) == 1).all() and (di.sum(1) == 1).all()):
            continue
        transformed = DIRECTIONS * np.where(flips, -1, 1)
        normal = (transformed @ DIRECTIONS.T).argmax(1)
        offset = DIRECTIONS @ (2 * aperture_center * flips)
        variants.append((ai.argmax(1), di.argmax(1), normal, offset))
    return variants


def _support(mask):
    if not np.any(mask):
        raise ValueError('Training shapes must occupy at least one voxel')
    # Maxima agree with those of the convex hull; boundary voxel centers suffice.
    xyz = np.argwhere(mask & ~binary_erosion(mask))
    return np.max(xyz.astype('f4') @ DIRECTIONS.T, axis=0)


def _fit_projection(data, standardize, rng):
    mean = data.mean(0)
    scale = np.maximum(data.std(0), 1e-3) if standardize else np.ones(data.shape[1], dtype='f4')
    centered = np.clip((data - mean) / scale, -10, 10)
    rank = min(64, max(1, len(data) - 1), data.shape[1])
    # Randomized range finder avoids an archive-sized dense SVD.
    width = min(rank + 12, *centered.shape)
    sample = centered @ rng.standard_normal((centered.shape[1], width), dtype=np.float32)
    q = np.linalg.qr(sample, mode='reduced')[0]
    q = np.linalg.qr(centered @ (centered.T @ q), mode='reduced')[0]
    reduced = q.T @ centered
    eig, left = np.linalg.eigh(reduced @ reduced.T)
    index = np.argsort(eig)[::-1][:rank]
    singular = np.sqrt(np.maximum(eig[index], 1e-12))
    basis = (reduced.T @ left[:, index]) / singular
    # Partial whitening retains signal amplitude while leaving room for weaker RF modes.
    basis /= np.maximum(singular / np.sqrt(len(data)), 0.025 * singular[0] / np.sqrt(len(data))) ** 0.35
    norm = max(float(np.linalg.norm(centered @ basis) / np.sqrt(len(data))), 1e-6)
    basis /= norm
    return mean, scale, basis.astype('f4')


def _project(groups, transforms):
    return [np.ascontiguousarray(np.clip((group - mean) / scale, -10, 10) @ basis, dtype='f4')
            for group, (mean, scale, basis) in zip(groups, transforms)]


def _blend(groups, mix):
    return np.concatenate([g * np.sqrt(w) for g, w in zip(groups, mix)], axis=1).astype('f4')


def _dist(a, b):
    return np.maximum(np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1) - 2*(a @ b.T), 0.)


def _design(x, anchors, width):
    return np.concatenate((x, np.exp(-_dist(x, anchors) / width)), axis=1)


def _local(x, query, y, k, ridge):
    k = min(k, len(x))
    delta2 = _dist(query, x)
    indices = np.argpartition(delta2, k-1, axis=1)[:, :k]
    outputs = []
    for q, neighbors in zip(query, indices):
        diff = (x[neighbors] - q).astype('f8')
        gram = diff @ diff.T
        gram.flat[::k+1] += ridge * max(np.trace(gram) / k, 1e-8)
        w = cho_solve(cho_factor(gram, check_finite=False), np.ones(k), check_finite=False)
        outputs.append((w / w.sum()) @ y[neighbors])
    return np.asarray(outputs, dtype='f4')


def _hull_box(support, shape, padding):
    lo = np.maximum(np.ceil(-support[[1, 3, 5]] - padding - 1e-5).astype(int), 0)
    hi = np.minimum(np.floor(support[[0, 2, 4]] + padding + 1e-5).astype(int) + 1, shape)
    slices = tuple(slice(int(i), int(j)) for i, j in zip(lo, np.maximum(lo, hi)))
    if (hi <= lo).any():
        return slices, np.empty(tuple(np.maximum(0, hi-lo)), dtype='f4')
    xyz = np.moveaxis(np.indices(tuple(hi-lo), dtype='f4'), 0, -1) + lo.astype('f4')
    # Process planes separately to bound peak memory for large output grids.
    margin = np.stack([(plane @ DIRECTIONS.T - support).max(axis=-1) for plane in xyz])
    return slices, margin


def _mask(support, shape, offset):
    result = np.zeros(shape, dtype=bool)
    slices, margin = _hull_box(support, shape, offset)
    result[slices] = margin <= offset + 1e-5
    return result


def _sdf(mask, support, offset):
    if not mask.any():
        # An out-of-volume or subvoxel hull has no occupied centers. Return its
        # positive supporting-plane distances so callers still get finite values.
        axes = np.indices(mask.shape, dtype='f4').transpose(1, 2, 3, 0)
        return np.stack([np.maximum((plane @ DIRECTIONS.T - support).max(-1) - offset, 1e-5)
                         for plane in axes]).astype('f4')
    return occupancy_sdf(mask)


def occupancy_sdf(mask):
    """Signed cell-center distances to the boundary of the occupied voxel union."""
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != bool:
        raise ValueError("Expected a three-dimensional Boolean mask")
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype='f4')
    # A center's closest point on an exposed unit voxel face falls on the
    # half-cell lattice. Dilation identifies that lattice's shared air/solid boundary.
    padded = np.pad(mask, 1)
    fine_shape = tuple(2 * n - 1 for n in padded.shape)
    solid = np.zeros(fine_shape, dtype=bool)
    air = np.zeros_like(solid)
    solid[::2, ::2, ::2] = padded
    air[::2, ::2, ::2] = ~padded
    cube = np.ones((3, 3, 3), dtype=bool)
    boundary = binary_dilation(solid, cube) & binary_dilation(air, cube)
    distance = distance_transform_edt(~boundary)[2:-2:2, 2:-2:2, 2:-2:2] * .5
    distance[mask] *= -1
    return distance.astype('f4')


def _validation_iou(predicted, masks, offsets):
    scores = np.zeros(len(offsets))
    for h, truth in zip(predicted, masks):
        slices, margin = _hull_box(h, truth.shape, max(offsets))
        local_truth = truth[slices]
        for j, offset in enumerate(offsets):
            result = margin <= offset + 1e-5
            common = np.count_nonzero(local_truth & result)
            union = np.count_nonzero(truth) + np.count_nonzero(result) - common
            scores[j] += common / union if union else 1.
    return scores / len(masks)


def fit(data, output, *, reflections=True, representation='voltage'):
    rng = np.random.default_rng(734)
    with _archive(data) as f:
        settings = _settings(f)
        train = np.asarray(f['splits/train'], dtype=int)
        validation = np.asarray(f['splits/validation'], dtype=int)
        if not len(train) or not len(validation) or np.intersect1d(train, validation).size:
            raise ValueError('Fitting requires separate, nonempty training and validation splits')
        variants = _reflections(settings)
        if not reflections:
            variants = variants[:1]
        feature_rows, targets, templates = [], [], []
        for i in train:
            wave = _signal(f, i, settings)
            templates.append(wave.ravel() / max(np.linalg.norm(wave), 1e-12))
            support = _support(_target(f, i) < 0)
            for antennas, beams, normals, translation in variants:
                feature_rows.append(_echo_features(wave[antennas][:, beams], representation, settings['acquisition']))
                targets.append(support[normals] + translation)
        masks = [_target(f, i) < 0 for i in validation]
        query_features, template_ious = [], []
        templates = np.stack(templates)
        for i, truth in zip(validation, masks):
            wave = _signal(f, i, settings)
            query_features.append(_echo_features(wave, representation, settings['acquisition']))
            neighbor = train[np.argmax(templates @ wave.ravel())]
            proposal = _target(f, neighbor) < 0
            template_ious.append(np.count_nonzero(proposal & truth) / max(1, np.count_nonzero(proposal | truth)))
        scene_ids = f['scene_ids'].asstr()
        training_ids = [scene_ids[i] for i in train]
        validation_ids = [scene_ids[i] for i in validation]
    del templates
    targets = np.asarray(targets, dtype='f4')
    groups = [np.stack(block) for block in zip(*feature_rows)]
    del feature_rows
    val_groups = [np.stack(block) for block in zip(*query_features)]
    transforms = [_fit_projection(g, index == 2, rng) for index, g in enumerate(groups)]
    compressed, queries = _project(groups, transforms), _project(val_groups, transforms)
    del groups, val_groups
    offsets = (-0.75, -0.25, 0.25, 0.75)
    best, winner, parameters = -1., None, None

    def score(predictions, config, arrays):
        nonlocal best, winner, parameters
        ious = _validation_iou(predictions, masks, offsets)
        j = int(np.argmax(ious))
        if ious[j] > best:
            best = float(ious[j])
            winner = dict(config, offset=offsets[j])
            parameters = arrays

    pools = [1, len(variants)] if len(variants) > 1 else [1]
    for n_variants in pools:
        take = np.arange(0, len(targets), len(variants)) if n_variants == 1 else np.arange(len(targets))
        y = targets[take]
        for mix in ((.45, .30, .25), (.15, .25, .60)):
            x = _blend([g[take] for g in compressed], mix)
            q = _blend(queries, mix)
            config = dict(mix=list(mix), reflections=n_variants)
            for k, ridge in ((12, .05), (40, .02), (56, .15)):
                predictions = _local(x, q, y, k, ridge)
                score(predictions, dict(config, method='local', k=k, ridge=ridge),
                      dict(training=x, supports=y))
            # Ridge regression combines a linear feature map and radial basis
            # functions centered on a fixed training-only sample.
            index = np.random.default_rng(19).choice(len(x), min(192, len(x)), replace=False)
            anchors = x[index]
            center_distance = _dist(anchors, anchors)
            positive = center_distance[center_distance > 1e-8]
            median = float(np.median(positive)) if len(positive) else 1.
            target_mean = y.mean(0)
            for width_factor in (.45, 1.5):
                width = max(1e-5, median * width_factor)
                design, evaluate = _design(x, anchors, width), _design(q, anchors, width)
                mean = design.mean(0)
                z = (design - mean).astype('f8')
                gram = z.T @ z
                cross = z.T @ (y - target_mean)
                for ridge in (.01, .12):
                    penalty = ridge * max(np.trace(gram) / len(gram), 1e-5)
                    regularized = gram.copy()
                    regularized.flat[::len(gram)+1] += penalty
                    coef = cho_solve(cho_factor(regularized, check_finite=False), cross,
                                     check_finite=False).astype('f4')
                    predictions = (evaluate - mean) @ coef + target_mean
                    score(predictions, dict(config, method='ridge', width=width, ridge=ridge),
                          dict(anchors=anchors, design_mean=mean, coef=coef, target_mean=target_mean))
    meta = dict(format_version=1 if representation == 'voltage' else 2, representation=representation,
                settings=settings, choice=winner, training_ids=training_ids,
                validation_ids=validation_ids, validation_mean_scene_iou=best,
                validation_template_mean_scene_iou=float(np.mean(template_ious)))
    payload = dict(parameters)
    for i, (mean, scale, basis) in enumerate(transforms):
        payload.update({f'mean_{i}': mean, f'scale_{i}': scale, f'basis_{i}': basis})
    output = Path(output)
    _save_model(output, meta, payload)
    return dict(representation=representation, training_scenes=len(train), validation_scenes=len(validation), choice=winner,
                validation_mean_scene_iou=best, validation_template_mean_scene_iou=float(np.mean(template_ious)),
                model_bytes=output.stat().st_size)


def _save_model(output, metadata, arrays):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix='.echo-', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, meta=np.array(json.dumps(metadata, allow_nan=False)), **arrays)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


class EchoKernel:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as store:
            self.meta = json.loads(store['meta'].item())
            self.arrays = {k: store[k] for k in store.files if k != 'meta'}
        version = self.meta.get('format_version')
        self.representation = self.meta.get('representation', 'voltage' if version == 1 else None)
        if (version not in (1, 2) or self.representation not in REPRESENTATIONS or
                (version == 1 and self.representation != 'voltage')):
            raise ValueError('Unknown EchoKernel model format')
        self.settings = self.meta['settings']
        self.transforms = [tuple(self.arrays[f'{key}_{i}'] for key in ('mean', 'scale', 'basis'))
                           for i in range(3)]

    def predict(self, wave):
        """Return a full-resolution signed distance field in cells from a calibrated archive waveform."""
        if tuple(wave.shape) != tuple(self.settings['sensor_shape']):
            raise ValueError('RF input axes differ from those used in training')
        groups = [g[None] for g in _echo_features(wave, self.representation, self.settings['acquisition'])]
        choice = self.meta['choice']
        q = _blend(_project(groups, self.transforms), choice['mix'])
        p = self.arrays
        if choice['method'] == 'ridge':
            design = _design(q, p['anchors'], choice['width'])
            support = ((design - p['design_mean']) @ p['coef'] + p['target_mean'])[0]
        else:
            support = _local(p['training'], q, p['supports'], choice['k'], choice['ridge'])[0]
        mask = _mask(support, self.settings['grid_shape'], choice['offset'])
        return _sdf(mask, support, choice['offset'])

    def check_archive(self, f):
        if _settings(f) != self.settings:
            raise ValueError('Archive preprocessing or acquisition differs from the fitted model')

    def predict_scene(self, directory):
        """Predict directly from a scene folder containing the RF pair and acquisition metadata."""
        return self.predict(_calibrated_pair(directory, self.settings))


def _calibrated_pair(directory, settings):
    directory = Path(directory)
    acquisition = json.loads((directory / 'acquisition.json').read_text())
    if acquisition != settings['acquisition']:
        raise ValueError('Raw scene acquisition differs from the fitted model')
    raw = np.load(directory / 'sensor_samples.npy', allow_pickle=False).astype('f8')
    empty = np.load(directory / 'empty_sensor_samples.npy', allow_pickle=False).astype('f8')
    if raw.shape != empty.shape:
        raise ValueError('Measured and empty recordings have different shapes')
    layout = acquisition.get('sensor_layout')
    if layout == 'direction,antenna,timestep':
        raw, empty = raw.transpose(1, 0, 2), empty.transpose(1, 0, 2)
    elif layout != 'antenna,direction,timestep':
        raise ValueError('Raw inference needs explicit 3-D sensor axes')
    wave = (raw - empty - settings['sensor_offset']) / settings['sensor_scale']
    return wave.astype('f4')


def evaluate(model, data, split):
    rows = []
    with _archive(data) as f:
        model.check_archive(f)
        indices = np.asarray(f[f'splits/{split}'], dtype=int)
        if not len(indices):
            raise ValueError(f'No scenes in {split} split')
        for i in indices:
            predicted = model.predict(_signal(f, i, model.settings))
            true = _target(f, i)
            occupied, target = predicted < 0, true < 0
            intersection, union = int(np.count_nonzero(occupied & target)), int(np.count_nonzero(occupied | target))
            rows.append(dict(scene=f['scene_ids'].asstr()[i], occupancy_iou=intersection / union if union else 1.,
                             occupied_intersection=intersection, occupied_union=union,
                             predicted_occupied=int(occupied.sum()), true_occupied=int(target.sum()),
                             sdf_mae_cells=float(np.mean(np.abs(predicted - true)))))
    total = sum(r['occupied_union'] for r in rows)
    return dict(scenes=rows, summary=dict(split=split, scenes=len(rows),
                mean_scene_iou=float(np.mean([r['occupancy_iou'] for r in rows])),
                pooled_iou=sum(r['occupied_intersection'] for r in rows) / total if total else 1.,
                mean_sdf_mae_cells=float(np.mean([r['sdf_mae_cells'] for r in rows]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    fit_args = commands.add_parser('fit')
    fit_args.add_argument('--data', type=Path, required=True)
    fit_args.add_argument('--output', type=Path, required=True)
    fit_args.add_argument('--no-reflections', action='store_true', help='Fit using only measured training views')
    fit_args.add_argument('--representation', choices=REPRESENTATIONS, default='voltage',
                          help='Use signed voltage, coherent baseband I/Q, or baseband envelope magnitude')
    eval_args = commands.add_parser('evaluate')
    eval_args.add_argument('--data', type=Path, required=True)
    eval_args.add_argument('--model', type=Path, required=True)
    eval_args.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    eval_args.add_argument('--output', type=Path)
    infer = commands.add_parser('infer')
    inputs = infer.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data', type=Path)
    inputs.add_argument('--scene-dir', type=Path)
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--scene-id')
    infer.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'fit':
        report = fit(args.data, args.output, reflections=not args.no_reflections, representation=args.representation)
    elif args.action == 'evaluate':
        report = evaluate(EchoKernel(args.model), args.data, args.split)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    else:
        model = EchoKernel(args.model)
        if args.scene_dir is not None:
            if args.scene_id:
                parser.error('--scene-id goes with --data')
            prediction = model.predict_scene(args.scene_dir)
        else:
            if not args.scene_id:
                parser.error('--data requires --scene-id')
            with _archive(args.data) as f:
                model.check_archive(f)
                ids = f['scene_ids'].asstr()[:].tolist()
                if args.scene_id not in ids:
                    parser.error('Unknown scene ID in archive')
                prediction = model.predict(_signal(f, ids.index(args.scene_id), model.settings))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as handle:
            np.save(handle, prediction)
        report = dict(output=str(args.output), shape=list(prediction.shape), units='grid cells')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
