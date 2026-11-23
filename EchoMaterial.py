"""Estimate a homogeneous laboratory inclusion's permittivity and conductivity from calibrated RF alone."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import label

from EchoKernel import (
    _archive, _blend, _calibrated_pair, _design, _dist, _echo_features, _fit_projection,
    _project, _save_model, _settings, _signal,
)


PARAMETERS = ('relative_permittivity', 'conductivity_s_per_m')
RF_MIXES = ((.35, .15, .30, .20), (.10, .10, .30, .50), (.15, .05, .10, .70))


def _homogeneous(values, name):
    if not len(values):
        raise ValueError(f'{name} is absent from the labeled volume')
    low, high = values.min(0), values.max(0)
    if not np.allclose(low, high, rtol=2e-6, atol=[2e-5, 1e-7]):
        raise ValueError(f'{name} contains multiple electromagnetic materials')
    return (low + high) / 2


def _label(f, i):
    grid = f['grids'][i].astype('f8') * f.attrs['grid_scale'] + f.attrs['grid_min']
    if not np.isfinite(grid).all():
        raise ValueError('Nonfinite material or signed-distance labels')
    solid = grid[..., 2] < 0
    if label(solid, structure=np.ones((3, 3, 3), dtype=int))[1] != 1:
        raise ValueError('The material estimator requires a single connected inclusion')
    return _homogeneous(grid[solid, :2], 'Inclusion'), _homogeneous(grid[~solid, :2], 'Background')


def _features(wave, acquisition):
    wave = np.asarray(wave, dtype='f4')
    signed, unit, spatial = _echo_features(wave)
    steps = wave.shape[-1]
    # Restore signed amplitude removed by cosine normalization. An archive's
    # training-fitted voltage scale is constant across all of its scenes.
    edges = np.linspace(0, steps, min(48, steps) + 1, dtype=int)
    pooled = np.add.reduceat(wave, edges[:-1], axis=-1) / np.diff(edges)
    traces = wave.reshape(-1, steps).astype('f8')
    power = traces**2
    total = power.sum(1)
    norm = float(np.linalg.norm(wave))
    floor = max(float(np.sqrt(total.max())) * 1e-6, 1e-12)
    rms = np.sqrt(total / steps)
    freq = np.fft.rfftfreq(steps)
    dt = acquisition.get('time_step_s', 0.) * acquisition.get('sample_stride', 1)
    carrier = acquisition.get('frequency_hz', 0.) * dt
    if carrier > 0:
        frequency_bins = np.abs(freq[:, None] - carrier * np.linspace(.25, 2.5, 12)).argmin(0)
    else:
        frequency_bins = np.linspace(1, max(1, len(freq)//2), 12).astype(int).clip(max=len(freq)-1)
    spectrum = np.abs(np.fft.rfft(traces, axis=-1)[:, frequency_bins])
    spectral_shape = np.log(np.maximum(spectrum, floor) / np.maximum(np.sqrt(total[:, None]), floor))
    cdf = np.cumsum(power, axis=-1)
    first = np.argmax(cdf >= .05 * total[:, None], axis=-1) / max(steps-1, 1)
    mid = np.argmax(cdf >= .5 * total[:, None], axis=-1) / max(steps-1, 1)
    late = np.argmax(cdf >= .9 * total[:, None], axis=-1) / max(steps-1, 1)
    stats = np.column_stack([np.log(np.maximum(rms, floor)), traces.max(1), traces.min(1),
                             first, mid, late-mid, *spectral_shape.T])
    # Sort by signal strength to retain near/far port differences independently
    # of which side of the aperture sees the strongest return.
    descending = np.argsort(-rms)
    groups = np.array_split(descending, min(8, len(traces)))
    summaries = np.stack([stats[ids].mean(0) for ids in groups])
    aggregate = np.r_[np.log(max(norm, 1e-12)), np.log(max(rms.max(), floor)),
                      summaries.ravel(), np.quantile(stats, [.1, .5, .9], axis=0).ravel()]
    return (pooled.ravel().astype('f4'), np.r_[signed, unit].astype('f4'),
            np.r_[spatial, stats.ravel(), aggregate].astype('f4'), aggregate.astype('f4'))


def _rows(f, indices, settings):
    rows, targets, backgrounds = [], [], []
    for i in indices:
        target, background = _label(f, i)
        rows.append(_features(_signal(f, i, settings), settings['acquisition']))
        targets.append(target)
        backgrounds.append(background)
    return [np.stack(group) for group in zip(*rows)], np.stack(targets), np.stack(backgrounds)


def _physical(values):
    return np.maximum(values, [1e-6, 0.])


def _fit_regressor(train_features, validation_features, y, truth, standardize, mixes):
    rng = np.random.default_rng(127)
    transforms = [_fit_projection(group, standardize[i], rng) for i, group in enumerate(train_features)]
    xf = _project(train_features, transforms)
    qf = _project(validation_features, transforms)
    mean = y.mean(0)
    spread = y.std(0)
    scale = np.where(spread > 1e-10, spread, 1.)
    centered = (y - mean) / scale
    best = np.abs(truth - mean).mean(0) / scale
    choices = [dict(method='train_mean'), dict(method='train_mean')]
    learned = [{}, {}]

    # For scalar material parameters, a location-invariant spectral summary is
    # useful alongside antenna-resolved signed amplitudes and echo timing.
    for mix in mixes:
        x, query = _blend(xf, mix), _blend(qf, mix)
        indices = np.random.default_rng(37).choice(len(x), min(160, len(x)), replace=False)
        anchors = x[indices]
        dist = _dist(anchors, anchors)
        median = np.median(dist[dist > 1e-8]) if (dist > 1e-8).any() else 1.
        for width_factor in (0., .6, 1.8):
            width = float(median * width_factor)
            design = _design(x, anchors, width) if width else x
            evaluate = _design(query, anchors, width) if width else query
            design_mean = design.mean(0)
            z = (design - design_mean).astype('f8')
            gram, cross = z.T @ z, z.T @ centered
            for ridge in (.02, .25, 2.):
                penalized = gram.copy()
                penalized.flat[::len(gram)+1] += ridge * max(np.trace(gram) / len(gram), 1e-6)
                coef = cho_solve(cho_factor(penalized, check_finite=False), cross,
                                 check_finite=False).astype('f4')
                prediction = _physical(((evaluate - design_mean) @ coef) * scale + mean)
                for shrink in (.55, 1.):
                    errors = np.abs(truth - (mean + shrink * (prediction - mean))).mean(0) / scale
                    for j in range(2):
                        if errors[j] < best[j]:
                            best[j] = errors[j]
                            choices[j] = dict(method='ridge', mix=list(mix), width=width,
                                              ridge=ridge, shrink=shrink)
                            learned[j] = dict(design_mean=design_mean, coef=coef[:, j])
                            if width:
                                learned[j]['anchors'] = anchors
    arrays = {}
    for i, (m, s, b) in enumerate(transforms):
        arrays.update({f'mean_{i}': m, f'scale_{i}': s, f'basis_{i}': b})
    for j, group in enumerate(learned):
        arrays.update({f'{key}_p{j}': value for key, value in group.items()})
    details = dict(choices=choices, train_mean=mean.tolist(), train_spread=spread.tolist(),
                   target_scale=scale.tolist(), validation_normalized_mae=best.tolist())
    return details, arrays


def fit(data, output):
    with _archive(data) as f:
        settings = _settings(f)
        training = np.asarray(f['splits/train'], dtype=int)
        valid = np.asarray(f['splits/validation'], dtype=int)
        if not len(training) or not len(valid) or np.intersect1d(training, valid).size:
            raise ValueError('Separate nonempty training and validation splits are required')
        train_features, y, bg = _rows(f, training, settings)
        validation_features, truth, val_background = _rows(f, valid, settings)
        names = f['scene_ids'].asstr()
        training_ids = [names[i] for i in training]
        validation_ids = [names[i] for i in valid]
    background = _homogeneous(bg, 'Training background')
    if not np.allclose(val_background, background, rtol=2e-6, atol=[2e-5, 1e-7]):
        raise ValueError('Training and validation background materials disagree')
    details, arrays = _fit_regressor(train_features, validation_features, y, truth, (True, False, True, True), RF_MIXES)
    details.update(model_type='echo_material', format_version=1, settings=settings, training_ids=training_ids,
                   validation_ids=validation_ids, background=background.tolist())
    best = np.array(details['validation_normalized_mae'])
    choices = details['choices']
    _save_model(output, details, arrays)
    return dict(training_scenes=len(training), validation_scenes=len(valid),
                validation_normalized_mae=dict(zip(PARAMETERS, best.tolist())),
                validation_mean_normalized_mae=float(best.mean()), choices=dict(zip(PARAMETERS, choices)),
                model_bytes=Path(output).stat().st_size)


def _predict_features(groups, transforms, meta, arrays):
    codes = _project([g[None] for g in groups], transforms)
    mean = np.asarray(meta['train_mean'])
    result = mean.copy()
    for j, choice in enumerate(meta['choices']):
        if choice['method'] != 'train_mean':
            x = _blend(codes, choice['mix'])
            if choice['width']:
                x = _design(x, arrays[f'anchors_p{j}'], choice['width'])
            value = float(((x - arrays[f'design_mean_p{j}']) @ arrays[f'coef_p{j}']).item())
            absolute = max(1e-6 if j == 0 else 0., mean[j] + value * meta['target_scale'][j])
            result[j] += choice['shrink'] * (absolute - mean[j])
    return result


class EchoMaterial:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as store:
            self.meta = json.loads(store['meta'].item())
            self.arrays = {k: store[k] for k in store.files if k != 'meta'}
        if self.meta.get('model_type') != 'echo_material' or self.meta.get('format_version') != 1:
            raise ValueError('Expected an EchoMaterial model')
        self.settings = self.meta['settings']
        self.transforms = [tuple(self.arrays[f'{key}_{i}'] for key in ('mean', 'scale', 'basis'))
                           for i in range(4)]

    def check_archive(self, f):
        if _settings(f) != self.settings:
            raise ValueError('Archive preprocessing or acquisition differs from the fitted material model')

    def predict(self, wave):
        """Return relative permittivity and conductivity (S/m), using only calibrated RF."""
        if tuple(wave.shape) != tuple(self.settings['sensor_shape']):
            raise ValueError('RF input axes differ from those used in training')
        groups = _features(wave, self.settings['acquisition'])
        return _predict_features(groups, self.transforms, self.meta, self.arrays)

    def predict_scene(self, directory):
        return self.predict(_calibrated_pair(directory, self.settings))

    def compose(self, estimate, predicted_sdf):
        """Compose a predicted geometry with the RF-derived inclusion values and training background."""
        sdf = np.asarray(predicted_sdf)
        if list(sdf.shape) != self.settings['grid_shape'] or not np.isfinite(sdf).all():
            raise ValueError('Expected a finite predicted SDF in cells on the fitted grid')
        return np.where((sdf < 0)[..., None], estimate, self.meta['background']).astype('f4')


def _scores(truth, predictions, scales):
    difference = predictions - truth
    variance = ((truth - truth.mean(0))**2).mean(0)
    mse = (difference**2).mean(0)
    mae = np.abs(difference).mean(0)
    result = {key: dict(mae=float(mae[j]), rmse=float(np.sqrt(mse[j])),
                        r2=(float(1-mse[j]/variance[j]) if variance[j] > 1e-16 else None))
              for j, key in enumerate(PARAMETERS)}
    result['mean_normalized_mae'] = float(np.mean(mae / scales))
    return result


def evaluate(model, data, split):
    ids, truth, predictions = [], [], []
    with _archive(data) as f:
        model.check_archive(f)
        for i in f[f'splits/{split}'][:]:
            material, unused_background = _label(f, i)
            truth.append(material)
            predictions.append(model.predict(_signal(f, i, model.settings)))
            ids.append(f['scene_ids'].asstr()[i])
    if not ids:
        raise ValueError(f'No scenes in {split} split')
    truth, predictions = np.stack(truth), np.stack(predictions)
    scales = np.asarray(model.meta['target_scale'])
    mean = np.broadcast_to(model.meta['train_mean'], truth.shape)
    rows = [dict(scene=name, truth=dict(zip(PARAMETERS, target.tolist())),
                 predicted=dict(zip(PARAMETERS, prediction.tolist())),
                 absolute_error=dict(zip(PARAMETERS, np.abs(prediction-target).tolist())))
            for name, target, prediction in zip(ids, truth, predictions)]
    # A cyclic reassignment after shuffling gives a reproducible RF control with
    # no correctly paired rows. The split's value distribution is preserved.
    shuffled = None
    if len(ids) > 1:
        permutation = np.random.default_rng(1741).permutation(len(ids))
        source = np.empty_like(permutation)
        source[permutation] = np.roll(permutation, 1)
        shuffled = _scores(truth, predictions[source], scales)
        for row, index in zip(rows, source):
            row['shuffled_rf_source'] = ids[index]
    return dict(scenes=rows, summary=dict(split=split, scenes=len(rows), matched_rf=_scores(truth, predictions, scales),
                train_mean=_scores(truth, mean, scales), shuffled_rf=shuffled, shuffle_seed=1741))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    fit_args = commands.add_parser('fit')
    fit_args.add_argument('--data', type=Path, required=True)
    fit_args.add_argument('--output', type=Path, required=True)
    ev = commands.add_parser('evaluate')
    ev.add_argument('--model', type=Path, required=True)
    ev.add_argument('--data', type=Path, required=True)
    ev.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    ev.add_argument('--output', type=Path)
    inf = commands.add_parser('infer')
    inf.add_argument('--model', type=Path, required=True)
    inputs = inf.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data', type=Path)
    inputs.add_argument('--scene-dir', type=Path)
    inf.add_argument('--scene-id')
    inf.add_argument('--predicted-sdf', type=Path, help='Previously predicted SDF in grid cells')
    inf.add_argument('--field-output', type=Path, help='NumPy volume with final channels [relative permittivity, S/m]')
    inf.add_argument('--output', type=Path, help='JSON file for the two scalar material estimates')
    args = parser.parse_args()
    if args.action == 'fit':
        report = fit(args.data, args.output)
    elif args.action == 'evaluate':
        report = evaluate(EchoMaterial(args.model), args.data, args.split)
    else:
        if (args.predicted_sdf is None) != (args.field_output is None):
            parser.error('--predicted-sdf and --field-output must be supplied together')
        model = EchoMaterial(args.model)
        if args.scene_dir:
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
        report = dict(zip(PARAMETERS, prediction.tolist()))
        if args.field_output:
            field = model.compose(prediction, np.load(args.predicted_sdf, allow_pickle=False))
            args.field_output.parent.mkdir(parents=True, exist_ok=True)
            with args.field_output.open('wb') as handle:
                np.save(handle, field)
            report['field_output'] = str(args.field_output)
    if args.action != 'fit' and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
