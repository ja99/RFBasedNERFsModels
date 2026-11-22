import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
from scipy.ndimage import label
import torch

from EchoSet import C, NORMALS, SetPredictor, _score, _slot_masks, _travel_bank, assignment_loss, components, evaluate, fit


class EchoSetTests(unittest.TestCase):
    def test_two_leg_vacuum_delay_and_empty_geometry_fallback(self):
        settings = dict(sensor_shape=[1, 1, 24], acquisition=dict(
            antenna_positions_cells=[[0, 0, 0]], transmit_positions_cells=[[0, 0, 0]],
            grid_origin_cells=[0, 0, 0], frequency_hz=2.4e9, grid_spacing_m=C*1e-12,
            time_step_s=1e-12, sample_stride=1, beamforming_angles_deg=[[0., 0.]]))
        bank = _travel_bank(settings, np.array([[4., 0., 0.], [6.25, 0., 0.]]))
        self.assertEqual(bank[:, 0].argmax(), 8)
        np.testing.assert_allclose(bank[12:14, 1], [.5, .5], atol=1e-5)
        np.testing.assert_allclose(bank.sum(0), 1.)
        self.assertIsNone(_travel_bank(dict(settings, acquisition={}), np.zeros((2, 3))))

    def test_set_loss_is_invariant_to_object_and_slot_order(self):
        single = np.zeros((16, 14, 12), dtype=bool)
        single[2:5, 3:6, 2:5] = True
        dual = single.copy()
        dual[11:14, 8:12, 4:8] = True
        targets = torch.from_numpy(np.stack([components(single), components(dual)]))
        c, logr, ratio = targets[..., 1:4], targets[..., 4], targets[..., 5:]
        exist = torch.where(targets[..., 0]>0, 4., -4.)
        decoded = (c+.1, logr-.12, ratio+.08, torch.ones_like(ratio), exist)
        err = assignment_loss(decoded, targets)
        torch.testing.assert_close(err, assignment_loss(decoded, targets.flip(1)))
        torch.testing.assert_close(err, assignment_loss(tuple(v.flip(1) for v in decoded), targets))
        overlap = (np.array([[4,4,4], [4,4,4]], dtype='f4'), np.full((2, len(NORMALS)), 2., dtype='f4'),
                   np.array([1., .9]), .99)
        slots = _slot_masks(overlap, single.shape, 0.)[0]
        self.assertEqual(label(slots[0] | slots[1])[1], 1)

    def test_voxel_satellite_stays_in_the_instance_and_the_scored_foreground(self):
        mask = np.zeros((15, 14, 12), dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        mask[5, 5, 4] = True  # an edge contact with the first solid
        mask[10:12, 9:11, 8:10] = True
        targets = components(mask)
        np.testing.assert_array_equal(targets[:, 0], [1., 1.])
        volumes = 4*np.pi/3*np.exp(3*targets[:, 4])
        np.testing.assert_allclose(volumes, [28, 8], rtol=1e-6)
        without_satellite = mask.copy()
        without_satellite[5, 5, 4] = False
        truth = np.where(mask, -.5, .5)
        output = _score([truth], [np.where(without_satellite, -.5, .5)], ['example'])
        row = output['scenes'][0]
        self.assertAlmostEqual(row['occupancy_iou'], 35/36)
        self.assertEqual(row['target_components'], 3)
        self.assertEqual(row['physical_instances']['target_components'], 2)
        self.assertIn('2', output['summary']['by_target_instances'])

    def test_mixed_fit_and_unlabeled_raw_inference(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'dataset.hdf5'
            meta = dict(sensor_layout='antenna,direction,timestep')
            rf = []
            t = np.linspace(0, 2*np.pi, 32)
            ids = [f'fixture_{i}' for i in range(18)]
            with h5py.File(path, 'w') as f:
                f.attrs.update(waveform_mode='calibrated', preprocessing_id='set_fixture', sensor_shape=[2, 1, 32],
                               sensor_storage='antenna_direction,time', acquisition_json=json.dumps(meta),
                               sensor_offset=0., sensor_scale=3., grid_min=[0.,0.,0.], grid_scale=[1.,1.,1.])
                f.create_dataset('scene_ids', data=ids, dtype=h5py.string_dtype())
                grids = f.create_dataset('grids', (18, 16, 15, 14, 3), dtype='f4')
                for i in range(len(ids)):
                    x, y, z = 2+i%3, 2+i%4, 2+(i//4)%3
                    mask = np.zeros((16, 15, 14), dtype=bool)
                    mask[x:x+4, y:y+4, z:z+4] = True
                    if i%2:
                        mask[10:13, 8:12, 7:11] = True
                    sdf = np.where(mask, -.5, .5)
                    grids[i] = np.repeat(sdf[..., None], 3, axis=-1)
                    wave = np.stack([np.sin(t*(1+i%2)+x/10), np.cos(t+y/10)*(.4+z/5)])
                    rf.append(wave[:, None, :].astype('f4'))
                f.create_dataset('sensor_samples', data=np.stack(rf).reshape(-1, 2, 32))
                for key, rows in dict(train=range(12), validation=range(12,16), test=[16,17]).items():
                    f.create_dataset('splits/'+key, data=list(rows))
                f['sensor_samples'][16:] = np.nan
                f['grids'][16:] = np.nan
            checkpoint = root/'echo-set.pt'
            fit(root, checkpoint, epochs=2, val_every=1, batch_size=4, hidden=24, queries=12)
            pred = SetPredictor(checkpoint)
            self.assertEqual(pred.training_ids, ids[:12])
            report = evaluate(pred, root, 'validation', batch_size=2)
            self.assertSetEqual(set(report['summary']['by_target_components']), {'1','2'})
            self.assertEqual(report['summary']['scenes'], 4)
            for row in report['controls']['shuffled_rf']['scenes']:
                self.assertNotEqual(row['scene'], row['echo_scene'])
            for row in report['controls']['cosine_nearest_training_template']['scenes']:
                self.assertIn(row['template_scene'], ids[:12])
            raw = root/'unlabeled'
            raw.mkdir()
            (raw/'acquisition.json').write_text(json.dumps(meta))
            empty = np.full(rf[13].shape, 27.)
            np.save(raw/'sensor_samples.npy', empty+3*rf[13].astype('f8'))
            np.save(raw/'empty_sensor_samples.npy', empty)
            field, slots, centers, p_two = pred.predict_scene(raw)
            np.testing.assert_array_equal(field, pred.fields(rf[13][None])[0][0])
            np.testing.assert_array_equal(field < 0, slots.any(0))
            self.assertEqual(centers.shape, (len(slots), 3))
            self.assertGreaterEqual(p_two, 0.)
            self.assertLessEqual(p_two, 1.)


if __name__ == '__main__':
    unittest.main()
