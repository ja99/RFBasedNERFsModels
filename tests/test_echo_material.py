import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from EchoMaterial import EchoMaterial, _label, _scores, evaluate, fit


class EchoMaterialTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = dict(sensor_layout='antenna,direction,timestep')
        self.ids = [f's_{i}' for i in range(26)]
        rng = np.random.default_rng(23)
        properties = np.column_stack([rng.uniform(2.5, 6., len(self.ids)), rng.uniform(.01, .08, len(self.ids))])
        t = np.linspace(0, 2*np.pi, 32, endpoint=False)
        rf = []
        with h5py.File(self.root/'dataset.hdf5', 'w') as f:
            f.attrs.update(waveform_mode='calibrated', preprocessing_id='synthetic_fixture',
                           sensor_shape=[2, 1, 32], sensor_storage='antenna_direction,time',
                           acquisition_json=json.dumps(self.meta), sensor_scale=2., sensor_offset=0.,
                           grid_min=[1., 0., -1.], grid_scale=[6., .1, 2.])
            f.create_dataset('scene_ids', data=self.ids, dtype=h5py.string_dtype())
            grids = f.create_dataset('grids', (len(self.ids), 8, 8, 8, 3), dtype='f4')
            for i, (er, sigma) in enumerate(properties):
                signal = np.stack([er*np.sin(t)/6, sigma*8*np.cos(2*t)]).astype('f4')
                rf.append(signal[:, None])
                grid = np.zeros((8, 8, 8, 3))
                grid[..., 2] = 1.
                grid[2:5, 3:5, 2:6] = [(er-1)/6, sigma/.1, 0.]
                grids[i] = grid
            f.create_dataset('sensor_samples', data=np.stack(rf).reshape(-1, 2, 32))
            for key, ids in dict(train=range(18), validation=range(18, 24), test=[24, 25]).items():
                f.create_dataset('splits/'+key, data=list(ids))
            # A model selector must not require RF or labels from the sealed split.
            f['grids'][24:] = np.nan
            f['sensor_samples'][24:] = np.nan
        self.rf, self.properties = np.stack(rf), properties

    def test_material_labels_and_homogeneity_failures(self):
        with h5py.File(self.root/'dataset.hdf5', 'r+') as f:
            estimate, background = _label(f, 0)
            np.testing.assert_allclose(estimate, self.properties[0], rtol=2e-6)
            np.testing.assert_array_equal(background, [1., 0.])
            f['grids'][0, 2, 3, 2, 0] += .1
            with self.assertRaisesRegex(ValueError, 'multiple'):
                _label(f, 0)
            f['grids'][1, 0, 0, 0, 2] = 0.
            with self.assertRaisesRegex(ValueError, 'single connected'):
                _label(f, 1)

    def test_controls_raw_inference_and_predicted_mask(self):
        checkpoint = self.root/'material.npz'
        report = fit(self.root, checkpoint)
        self.assertEqual(report['training_scenes'], 18)
        model = EchoMaterial(checkpoint)
        result = evaluate(model, self.root, 'validation')
        summary = result['summary']
        self.assertLess(summary['matched_rf']['mean_normalized_mae'], summary['train_mean']['mean_normalized_mae'])
        self.assertLess(summary['matched_rf']['mean_normalized_mae'], summary['shuffled_rf']['mean_normalized_mae'])
        for row in result['scenes']:
            self.assertNotEqual(row['shuffled_rf_source'], row['scene'])
        raw = self.root/'recording'
        raw.mkdir()
        empty = np.full(self.rf[18].shape, 7.)
        np.save(raw/'empty_sensor_samples.npy', empty)
        np.save(raw/'sensor_samples.npy', empty + self.rf[18].astype('f8') * 2)
        (raw/'acquisition.json').write_text(json.dumps(self.meta))
        estimate = model.predict_scene(raw)
        np.testing.assert_allclose(estimate, model.predict(self.rf[18]))
        predicted_sdf = np.ones((8, 8, 8))
        predicted_sdf[5:7, 2:4, 1:4] = -1.
        composed = model.compose(estimate, predicted_sdf)
        np.testing.assert_allclose(composed[5, 2, 1], estimate, rtol=1e-6)
        np.testing.assert_array_equal(composed[0, 0, 0], [1., 0.])

    def test_metric_units_and_r_squared(self):
        truth = np.array([[3., .01], [5., .03]])
        scores = _scores(truth, np.array([[4., .02], [4., .02]]), np.array([1., .01]))
        self.assertEqual(scores['relative_permittivity']['mae'], 1.)
        self.assertAlmostEqual(scores['conductivity_s_per_m']['rmse'], .01)
        self.assertAlmostEqual(scores['conductivity_s_per_m']['r2'], 0.)
        self.assertAlmostEqual(scores['mean_normalized_mae'], 1.)
        self.assertIsNone(_scores(truth[:1], truth[:1], [1., .01])['relative_permittivity']['r2'])


if __name__ == '__main__':
    unittest.main()
