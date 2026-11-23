import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from DepthMaterial import DepthMaterial, _depth_features, score, train
from EchoKernel import fit as fit_geometry
from Reconstruct import reconstruct


class DepthMaterialTests(unittest.TestCase):
    def test_metric_frame_and_misses(self):
        camera = {'intrinsics': dict(width=7, height=5, fx=3.5, fy=3.5, cx=3, cy=2)}
        frame = np.zeros((5, 7), dtype='f4')
        blank = _depth_features(frame, camera)
        self.assertTrue(all(np.isfinite(group).all() and not group.any() for group in blank))
        frame[1:4, 2:5] = .1
        measured = _depth_features(frame, camera)
        self.assertEqual([g.shape for g in blank], [g.shape for g in measured])
        self.assertAlmostEqual(measured[1][0], 9 / 35)
        with self.assertRaisesRegex(ValueError, 'optical-axis depth'):
            _depth_features(-frame, camera)

    def test_blind_file_inference_and_shuffled_controls(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root/'frames'
            frames.mkdir()
            camera = {'intrinsics': dict(width=8, height=8, fx=4., fy=4., cx=3.5, cy=3.5)}
            (frames/'camera.json').write_text(json.dumps(camera))
            original = dict(grid_origin_cells=[0, 0, 0], grid_spacing_m=.0045, sensor_layout='antenna,direction,timestep', antenna_positions_cells=[[0,0,0], [0,0,2]],
                            transmit_positions_cells=[[0,1,0],[0,1,2]], beamforming_angles_deg=[[0,0],[0,30]])
            acquisition = dict(original, original_transmit_array_cells=original['transmit_positions_cells'],
                               receive_channel_indices=[0,1], acquired_beam_indices=[0], beamforming_angles_deg=[[0,0]])
            rng = np.random.default_rng(628)
            waves = rng.normal(size=(27,2,2,32)).astype('f4')
            with h5py.File(root/'dataset.hdf5', 'w') as f:
                f.attrs.update(waveform_mode='calibrated', preprocessing_id='fixture', sensor_shape=[2,1,32],
                               sensor_storage='antenna_direction,time', acquisition_json=json.dumps(acquisition),
                               sensor_scale=2., sensor_offset=0., grid_min=[1.,0.,-1.], grid_scale=[4.,1.,2.])
                ids = [f's_{i}' for i in range(len(waves))]
                f.create_dataset('scene_ids', data=ids, dtype=h5py.string_dtype())
                f.create_dataset('sensor_samples', data=waves[:,:,0,:])
                labels = np.zeros((len(ids), 4,4,4,3), dtype='f4')
                labels[...,2] = 1.
                for i, scene in enumerate(ids):
                    labels[i,1:3,1:3,1:3,:] = [.4 + .04*waves[i,0,0,0], .03, 0.]
                    folder = frames/scene
                    folder.mkdir()
                    frame = np.zeros((8,8), dtype='f4')
                    frame[1:5,2:6] = .15 + .001*i
                    np.save(folder/'depth.npy', frame)
                f.create_dataset('grids', data=labels)
                for split, indexes in dict(train=range(16), validation=range(16,22), test=range(22,27)).items():
                    f.create_dataset('splits/'+split, data=list(indexes))
                # Sealed grids can be missing at fit time.
                f['grids'][22:] = np.nan
            output = root/'models'
            train(root, frames, output)
            with h5py.File(root/'dataset.hdf5', 'r+') as f:
                f['grids'][22:] = labels[22:]
            summary = score(root, frames, output)
            rows = json.loads((output/'scenes.json').read_text())
            self.assertEqual(summary['test'], 5)
            self.assertTrue(all(row['shuffled_source'] != row['scene'] for row in rows))
            self.assertEqual(len({row['shuffled_source'] for row in rows}), len(rows))
            i = 22
            scene = root/'raw'
            scene.mkdir()
            (scene/'acquisition.json').write_text(json.dumps(original))
            # Only the acquired pair and calibration are present, with both original shots.
            empty = np.full_like(waves[i], 2., dtype='f8')
            np.save(scene/'empty_sensor_samples.npy', empty)
            np.save(scene/'sensor_samples.npy', empty + waves[i].astype('f8')*2)
            frame = frames/ids[i]/'depth.npy'
            for mode in ('rf', 'depth', 'fusion'):
                model = DepthMaterial(output/(mode+'.npz'))
                result = model.predict_files(scene if mode != 'depth' else None,
                                             frame if mode != 'rf' else None,
                                             frames/'camera.json' if mode != 'rf' else None)
                np.testing.assert_allclose(result, rows[0]['predictions'][mode], atol=1e-6)
            with self.assertRaisesRegex(ValueError, 'calibration differs'):
                model.predict(wave=waves[i,:,:1], depth=np.load(frame), camera={'intrinsics': {}})

            geometry = output/'geometry.npz'
            fit_geometry(root, geometry, reflections=False)
            options = dict(geometry_model=geometry, depth_material_model=output/'fusion.npz',
                           depth=frame, camera=frames/'camera.json')
            volume = root/'reconstructed.npz'
            reconstruct(scene, volume, **options)
            with np.load(volume, allow_pickle=False) as result:
                mask = result['occupied']
                self.assertTrue(mask.any())
                np.testing.assert_array_equal(mask, result['sdf_cells'] < 0)
                np.testing.assert_allclose(result['permittivity'][mask], rows[0]['predictions']['fusion'][0])
                np.testing.assert_allclose(result['conductivity_s_per_m'][mask], rows[0]['predictions']['fusion'][1])
                np.testing.assert_allclose(result['permittivity'][~mask], 1.)
                np.testing.assert_array_equal(result['geometry_beam_indices'], [0])
                np.testing.assert_array_equal(result['material_beam_indices'], [0])
            # Correcting visible geometry must also move the material-field boundary.
            free, hit = np.zeros_like(mask), np.zeros_like(mask)
            free[tuple(np.argwhere(mask)[0])] = True
            hit[tuple(np.argwhere(~mask)[0])] = True
            with patch('DepthGeometry.registered_camera') as optics:
                optics.return_value.observations.return_value = (free, hit)
                reconstruct(scene, volume, depth_geometry=True, depth_tolerance_mm=3, **options)
                self.assertAlmostEqual(optics.return_value.observations.call_args.kwargs['tolerance_m'], .003)
            with np.load(volume) as result:
                expected = (mask & ~free) | hit
                np.testing.assert_array_equal(result['occupied'], expected)
                np.testing.assert_array_equal(result['sdf_cells'] < 0, expected)
                self.assertAlmostEqual(float(result['sdf_cells'][hit][0]), -.5)
                self.assertTrue((result['permittivity'][free] == 1).all())
                np.testing.assert_allclose(result['permittivity'][hit], rows[0]['predictions']['fusion'][0])
            with self.assertRaisesRegex(ValueError, 'separate'):
                reconstruct(scene, frame, **options)
            with self.assertRaisesRegex(ValueError, 'learned compact single'):
                reconstruct(scene, volume, refine=True, **options)
