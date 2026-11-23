"""Estimate homogeneous inclusion material from calibrated RF, registered first-return depth, or both."""

import argparse
import json
from pathlib import Path

import numpy as np

from EchoKernel import _archive, _calibrated_pair, _save_model, _settings, _signal
from EchoMaterial import PARAMETERS, RF_MIXES, _features, _fit_regressor, _homogeneous, _label, _predict_features, _scores


# Every sensor configuration gets three feature mixtures and the same regularization search.
DEPTH_MIXES = ((.25, .75), (.50, .50), (.05, .95))
FUSION_MIXES = tuple(tuple(w * (1 - d) for w in rf) + (.15 * d, .85 * d)
                     for rf, d in zip(RF_MIXES, (.25, .45, .65)))
GROUPS = {'rf': (0, 1, 2, 3), 'depth': (4, 5), 'fusion': tuple(range(6))}
MIXES = {'rf': RF_MIXES, 'depth': DEPTH_MIXES, 'fusion': FUSION_MIXES}


def _depth_features(depth, camera):
    """Pool visible pixels and summarize surface position, scale, and silhouette in camera coordinates."""
    k = camera['intrinsics']
    z = np.asarray(depth, dtype=np.float64)
    if (z.shape != (k['height'], k['width']) or z.size == 0 or not np.isfinite(z).all()
            or np.any(z < 0) or min(k['fx'], k['fy']) <= 0):
        raise ValueError('Expected finite nonnegative optical-axis depth (metres) matching camera intrinsics')
    valid = z > 0
    # Each patch records occupancy and unconditioned mean optical depth, so a miss never supplies a surface.
    blocks = np.stack([valid.astype(float), z])
    by_row = np.stack([rows.mean(1) for rows in np.array_split(blocks, min(8, z.shape[0]), axis=1)], axis=1)
    pooled = np.stack([cols.mean(2) for cols in np.array_split(by_row, min(8, z.shape[1]), axis=2)], axis=2)
    rows, cols = np.nonzero(valid)
    if not len(rows):
        return pooled.ravel().astype('f4'), np.zeros(32, dtype='f4')
    distance = z[valid]
    xy = np.column_stack([(cols-k['cx'])/k['fx'], (rows-k['cy'])/k['fy']])
    points = np.column_stack([xy * distance[:, None], distance])
    # Approximate projected visible area in a plane perpendicular to the optical axis.
    area = np.sum(distance**2)/(k['fx']*k['fy'])
    quantiles = np.quantile(points, [.02, .25, .5, .75, .98], axis=0)
    centered = points - points.mean(0)
    cov = centered.T @ centered / len(points)
    lengths = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
    spans = np.ptp(points, axis=0)
    stats = np.r_[valid.mean(), np.log(area), np.log(distance.mean()), np.log(len(rows)), quantiles.ravel(),
                  points.mean(0), spans, lengths, xy.mean(0), np.ptp(xy, axis=0)]
    return pooled.ravel().astype('f4'), stats.astype('f4')


def calibrated_subset(scene, settings):
    """Apply archive calibration and any trained receive/shot subset to raw voltages."""
    expected = settings['acquisition']
    original = json.loads((scene / 'acquisition.json').read_text())
    if original == expected:
        return _calibrated_pair(scene, settings)
    if 'receive_channel_indices' not in expected or 'acquired_beam_indices' not in expected:
        raise ValueError('RF acquisition differs from training')
    rx, shots = expected['receive_channel_indices'], expected['acquired_beam_indices']
    converted = dict(original, original_transmit_array_cells=original['transmit_positions_cells'],
                     receive_channel_indices=rx, acquired_beam_indices=shots,
                     antenna_positions_cells=[original['antenna_positions_cells'][i] for i in rx],
                     beamforming_angles_deg=[original['beamforming_angles_deg'][i] for i in shots])
    if converted != expected:
        raise ValueError('RF acquisition differs from training after beam/channel selection')
    full = _calibrated_pair(scene, dict(settings, acquisition=original))
    if full.shape[:2] != (len(original['antenna_positions_cells']), len(original['beamforming_angles_deg'])):
        raise ValueError('Recorded channel or shot count differs from acquisition metadata')
    return full[rx][:, shots]


class DepthMaterial:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as store:
            self.meta = json.loads(store['meta'].item())
            self.arrays = {k: store[k] for k in store.files if k != 'meta'}
        if self.meta.get('model_type') != 'depth_material' or self.meta.get('format_version') != 1:
            raise ValueError('Expected a DepthMaterial checkpoint')
        self.mode = self.meta['mode']
        self.settings = self.meta['settings']
        self.transforms = [tuple(self.arrays[f'{key}_{i}'] for key in ('mean', 'scale', 'basis'))
                           for i in range(len(GROUPS[self.mode]))]

    def predict(self, *, wave=None, depth=None, camera=None):
        groups = []
        if self.mode != 'depth':
            if np.shape(wave) != tuple(self.settings['sensor_shape']):
                raise ValueError('Expected a calibrated waveform with the fitted acquisition axes')
            groups.extend(_features(wave, self.settings['acquisition']))
        if self.mode != 'rf':
            if camera != self.meta['camera']:
                raise ValueError('Depth camera calibration differs from training')
            groups.extend(_depth_features(depth, camera))
        return _predict_features(groups, self.transforms, self.meta, self.arrays)

    def predict_files(self, rf_scene=None, depth_file=None, camera_file=None):
        wave = depth = camera = None
        if self.mode != 'depth':
            if rf_scene is None:
                raise ValueError('This estimator needs an RF recording')
            wave = calibrated_subset(Path(rf_scene), self.settings)
        if self.mode != 'rf':
            if depth_file is None or camera_file is None:
                raise ValueError('This estimator needs a depth frame and camera calibration')
            depth = np.load(depth_file, allow_pickle=False)
            camera = json.loads(Path(camera_file).read_text())
        return self.predict(wave=wave, depth=depth, camera=camera)


def _observations(f, split, frames, camera, settings):
    indices = np.array(f[f'splits/{split}'], dtype=int)
    if not len(indices):
        raise ValueError(f'Empty {split} split')
    ids, rows = [], []
    for i in indices:
        scene = f['scene_ids'].asstr()[i]
        wave = _signal(f, i, settings)
        depth = np.load(frames / scene / 'depth.npy', allow_pickle=False)
        rows.append((*_features(wave, settings['acquisition']), *_depth_features(depth, camera)))
        ids.append(scene)
    return indices, ids, [np.stack(group) for group in zip(*rows)]


def _select(blocks, mode):
    return [blocks[j] for j in GROUPS[mode]]


def train(data, frames, output):
    camera = json.loads((frames / 'camera.json').read_text())
    with _archive(data) as f:
        settings = _settings(f)
        train_idx, train_ids, x = _observations(f, 'train', frames, camera, settings)
        val_idx, val_ids, v = _observations(f, 'validation', frames, camera, settings)
        if len(set(train_ids + val_ids)) != len(train_ids) + len(val_ids):
            raise ValueError('Training and validation scenes must be distinct')
        y, backgrounds = map(np.stack, zip(*(_label(f, i) for i in train_idx)))
        truth, val_backgrounds = map(np.stack, zip(*(_label(f, i) for i in val_idx)))
        background = _homogeneous(backgrounds, 'Training background')
        if not np.allclose(val_backgrounds, background, rtol=2e-6, atol=[2e-5, 1e-7]):
            raise ValueError('Training and validation backgrounds differ')
    for mode in GROUPS:
        standardize = _select((True, False, True, True, True, True), mode)
        meta, arrays = _fit_regressor(_select(x, mode), _select(v, mode), y, truth, standardize, MIXES[mode])
        meta.update(model_type='depth_material', format_version=1, settings=settings,
                    background=background.tolist(), camera=camera if mode != 'rf' else None, mode=mode, training_ids=train_ids, validation_ids=val_ids)
        _save_model(output / f'{mode}.npz', meta, arrays)


def score(data, frames, output):
    camera = json.loads((frames / 'camera.json').read_text())
    models = {mode: DepthMaterial(output / f'{mode}.npz') for mode in GROUPS}
    settings = models['fusion'].settings
    with _archive(data) as f:
        if any(m.settings != _settings(f) for m in models.values()):
            raise ValueError('Test RF acquisition or preprocessing differs from training')
        if any(m.meta['camera'] != camera for m in (models['fusion'], models['depth'])):
            raise ValueError('Test depth camera differs from training')
        indices, ids, groups = _observations(f, 'test', frames, camera, settings)
        for model in models.values():
            if set(ids).intersection(model.meta['training_ids'] + model.meta['validation_ids']):
                raise ValueError('A fitted scene reappeared in the test archive')
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError('Shuffled controls require at least two distinct test scenes')
        permutation = np.random.default_rng(1741).permutation(len(ids))
        source = np.empty_like(permutation)
        source[permutation] = np.roll(permutation, 1)
        predictions = {}
        for key, mode, shuffle in (('rf', 'rf', None), ('depth', 'depth', None), ('fusion', 'fusion', None),
                                  ('shuffled_rf', 'fusion', 'rf'), ('shuffled_depth', 'fusion', 'depth')):
            model = models[mode]
            selected = [g[source] if j in GROUPS.get(shuffle, ()) else g
                        for j, g in enumerate(groups) if j in GROUPS[mode]]
            predictions[key] = np.stack([_predict_features(row, model.transforms, model.meta, model.arrays)
                                         for row in zip(*selected)])
        # Evaluation labels become available after all matched and shuffled predictions.
        truth = np.stack([_label(f, i)[0] for i in indices])
    mean = models['fusion'].meta['train_mean']
    predictions['train_mean'] = np.broadcast_to(mean, truth.shape)
    scales = np.array(models['fusion'].meta['target_scale'])
    summary = dict(train=len(models['fusion'].meta['training_ids']), validation=len(models['fusion'].meta['validation_ids']),
                   test=len(ids), parameters=list(PARAMETERS), bootstrap_draws=10000, bootstrap_seed=9351,
                   shuffle_seed=1741, methods={}, contrasts={})
    draws = np.random.default_rng(summary['bootstrap_seed']).integers(len(ids), size=(summary['bootstrap_draws'],len(ids)))
    errors = {k: np.abs(v - truth) for k,v in predictions.items()}
    for method, prediction in predictions.items():
        metrics = _scores(truth, prediction, scales)
        ci = np.percentile(errors[method][draws].mean(1), [2.5, 97.5], axis=0)
        for j, prop in enumerate(PARAMETERS):
            metrics[prop]['mae_ci95'] = ci[:,j].tolist()
        summary['methods'][method] = metrics
    for comparator in ('rf', 'depth', 'shuffled_rf', 'shuffled_depth'):
        delta = errors['fusion'] - errors[comparator]
        ci = np.percentile(delta[draws].mean(1), [2.5, 97.5], axis=0)
        summary['contrasts']['fusion_minus_' + comparator] = {
            prop: dict(mean_mae_delta=float(delta[:,j].mean()), ci95=ci[:,j].tolist())
            for j, prop in enumerate(PARAMETERS)}
    rows = [dict(scene=s, truth=truth[i].tolist(), shuffled_source=ids[source[i]],
                 predictions={k: p[i].tolist() for k,p in predictions.items()}) for i,s in enumerate(ids)]
    for name, obj in [('summary',summary),('scenes',rows)]:
        (output / f'{name}.json').write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    study = commands.add_parser('study')
    for flag in ('train', 'test', 'frames', 'output'):
        study.add_argument('--' + flag, type=Path, required=True)
    infer = commands.add_parser('infer')
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--rf-scene', type=Path, help='Raw scene folder with voltage pair and acquisition.json')
    infer.add_argument('--depth', type=Path, help='Metric first-return depth.npy')
    infer.add_argument('--camera', type=Path, help='Registered camera.json')
    args = parser.parse_args()
    if args.action == 'study':
        train(args.train, args.frames, args.output)
        report = score(args.test, args.frames, args.output)
    else:
        values = DepthMaterial(args.model).predict_files(args.rf_scene, args.depth, args.camera)
        report = dict(zip(PARAMETERS, values.tolist()))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
