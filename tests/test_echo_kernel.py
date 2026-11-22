import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from EchoKernel import DIRECTIONS, EchoKernel, _mask, _reflections, _sdf, _support, evaluate, fit


class EchoKernelTests(unittest.TestCase):
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
            meta = dict(sensor_layout='antenna,direction,timestep')
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
            checkpoint = root / 'model.npz'
            summary = fit(root, checkpoint)
            model = EchoKernel(checkpoint)
            self.assertEqual(summary['training_scenes'], 7)
            self.assertEqual(model.meta['training_ids'], ids[:7])
            self.assertEqual(model.meta['validation_ids'], ids[7:9])
            scores = evaluate(model, root, 'validation')
            self.assertAlmostEqual(scores['summary']['mean_scene_iou'], summary['validation_mean_scene_iou'])
            raw = root / 'unlabeled'
            raw.mkdir()
            empty = np.full((2, 1, 32), 21.)
            np.save(raw/'empty_sensor_samples.npy', empty)
            np.save(raw/'sensor_samples.npy', 2*signals[7].astype('f8') + empty)
            (raw/'acquisition.json').write_text(json.dumps(meta))
            np.testing.assert_array_equal(model.predict_scene(raw), model.predict(signals[7]))
            (raw/'acquisition.json').write_text(json.dumps(dict(meta, frequency_hz=3e9)))
            with self.assertRaisesRegex(ValueError, 'acquisition'):
                model.predict_scene(raw)


if __name__ == '__main__':
    unittest.main()
