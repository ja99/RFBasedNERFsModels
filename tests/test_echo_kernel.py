import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from EchoKernel import DIRECTIONS, REPRESENTATIONS, EchoKernel, _baseband, _echo_features, _mask
from EchoKernel import _reflections, _sdf, _support, evaluate, fit


class EchoKernelTests(unittest.TestCase):
    def test_full_band_iq_remodulates_and_preserves_receiver_phase(self):
        metadata = dict(frequency_hz=2e9, time_step_s=1e-11, sample_stride=4)
        t = np.arange(256)
        phase = 2 * np.pi * .08 * t
        envelope = np.exp(-((t - 128) / 26.) ** 2)
        phases = np.array([0., .7])[:, None, None]
        pulse = envelope * np.cos(phase + phases)
        np.testing.assert_allclose(_baseband(pulse, metadata), envelope * np.exp(1j * phases), atol=1e-9)
        # Include DC, Nyquist and energy at the window edges.
        wave = np.random.default_rng(76).normal(size=(3, 2, 256))
        restored = (_baseband(wave, metadata) * np.exp(1j * phase)).real
        np.testing.assert_allclose(restored, wave, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'I/Q requires'):
            _baseband(wave, {})

    def test_voltage_and_envelope_controls_share_the_intended_groups(self):
        wave = np.random.default_rng(5).normal(size=(4, 1, 128))
        metadata = dict(frequency_hz=2.4e9, time_step_s=1e-11, sample_stride=2)
        voltage, iq, magnitude, control = [_echo_features(wave, r, metadata) for r in REPRESENTATIONS]
        for x, y in zip(voltage[:2], control[:2]):
            np.testing.assert_array_equal(x, y)
        for group in (magnitude, control):
            np.testing.assert_array_equal(iq[2], group[2])
        self.assertEqual(iq[0].size, 2 * magnitude[0].size)
        # Sign is an absolute phase rotation; the magnitude branch should lose it.
        for positive, negative in zip(magnitude, _echo_features(-wave, 'magnitude', metadata)):
            np.testing.assert_array_equal(positive, negative)
        np.testing.assert_array_equal(iq[0], -_echo_features(-wave, 'iq', metadata)[0])

    def test_exact_cuboid_support_and_signed_voxel_distances(self):
        target = np.zeros((9, 11, 12), dtype=bool)
        target[2:7, 3:8, 1:7] = True
        h = _support(target)
        np.testing.assert_array_equal(_mask(h, target.shape, 0.), target)
        field = _sdf(target, h, 0.)
        self.assertEqual(field[4, 5, 4], -2.5)
        self.assertEqual(field[1, 5, 4], 0.5)
        self.assertAlmostEqual(float(field[1, 2, 0]), np.sqrt(.75), places=6)

    def test_reflections_permute_beams_antennas_and_actual_supports(self):
        pos = [[4, y, z] for y in (2, 6) for z in (2, 6)]
        settings = dict(sensor_shape=[4, 5, 16], grid_shape=[9, 11, 12], acquisition=dict(
            antenna_positions_cells=pos, transmit_positions_cells=pos, grid_origin_cells=[0, 0, 0],
            beamforming_angles_deg=[[0, 0], [-30, 0], [30, 0], [0, -30], [0, 30]]))
        variants = _reflections(settings)
        self.assertEqual(len(variants), 4)
        ai, di, si, shift = variants[1]
        self.assertEqual(list(ai), [1, 0, 3, 2])
        self.assertEqual(list(di), [0, 1, 2, 4, 3])
        point = np.array([3, 2, 5])
        np.testing.assert_allclose((DIRECTIONS @ point)[si] + shift, DIRECTIONS @ [3, 2, 3], atol=1e-6)

    def test_training_avoids_test_rows_and_raw_inference_needs_no_grid(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'dataset.hdf5'
            ids = [f'scene_{i}' for i in range(10)]
            meta = dict(sensor_layout='antenna,direction,timestep', frequency_hz=2.4e9,
                        time_step_s=1e-11, sample_stride=2)
            t = np.arange(32)
            signals, grids = [], []
            for i in range(10):
                center = 2 + i % 6
                w = np.stack([np.sin((t-center-j)/3) * np.exp(-((t-2*center-j)/6)**2)
                              for j in (0, 1)]).reshape(2, 1, 32)
                signals.append(w.astype('f4'))
                mask = np.zeros((12, 9, 9), dtype=bool)
                mask[center-1:center+2, 2:5, 3:7] = True
                grids.append(_sdf(mask, _support(mask), 0.))
            with h5py.File(archive, 'w') as f:
                f.attrs.update(waveform_mode='calibrated', preprocessing_id='fixture', sensor_shape=[2, 1, 32],
                               sensor_storage='antenna_direction,time', acquisition_json=json.dumps(meta),
                               sensor_offset=0., sensor_scale=2., grid_min=[0., 0., 0.], grid_scale=[1., 1., 1.])
                f.create_dataset('scene_ids', data=ids, dtype=h5py.string_dtype())
                f.create_dataset('sensor_samples', data=np.stack(signals).reshape(10, 2, 32))
                f.create_dataset('grids', data=np.repeat(np.stack(grids)[..., None], 3, axis=-1))
                for name, indices in dict(train=range(7), validation=[7, 8], test=[9]).items():
                    f.create_dataset('splits/'+name, data=list(indices))
                # Training and validation must work without even finite values on the sealed split.
                f['sensor_samples'][9] = np.nan
                f['grids'][9] = np.nan
            raw = root / 'unlabeled'
            raw.mkdir()
            empty = np.full((2, 1, 32), 21.)
            np.save(raw/'empty_sensor_samples.npy', empty)
            np.save(raw/'sensor_samples.npy', 2*signals[7].astype('f8') + empty)
            (raw/'acquisition.json').write_text(json.dumps(meta))
            for representation in REPRESENTATIONS:
                with self.subTest(representation=representation):
                    checkpoint = root / (representation + '.npz')
                    summary = fit(root, checkpoint, representation=representation)
                    model = EchoKernel(checkpoint)
                    self.assertEqual(model.representation, representation)
                    self.assertEqual(summary['training_scenes'], 7)
                    self.assertEqual(model.meta['training_ids'], ids[:7])
                    self.assertEqual(model.meta['validation_ids'], ids[7:9])
                    scores = evaluate(model, root, 'validation')
                    self.assertAlmostEqual(scores['summary']['mean_scene_iou'], summary['validation_mean_scene_iou'])
                    np.testing.assert_array_equal(model.predict_scene(raw), model.predict(signals[7]))
            (raw/'acquisition.json').write_text(json.dumps(dict(meta, frequency_hz=3e9)))
            with self.assertRaisesRegex(ValueError, 'acquisition'):
                model.predict_scene(raw)


if __name__ == '__main__':
    unittest.main()
