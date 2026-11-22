import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from CalibrateReference import calibrate_pair, calibrate_scene, fit_direct_gains, prefix_lengths


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        rx = np.array([[6,4,4], [6,8,4], [6,4,8]])
        self.meta = dict(sensor_layout="antenna,direction,timestep", grid_origin_cells=[6,2,2],
                         grid_shape=[24,18,18], pml_thickness=2, antenna_positions_cells=rx.tolist(),
                         transmit_positions_cells=(rx + (0,1,0)).tolist(), grid_spacing_m=.0045,
                         time_step_s=8.58e-12, sample_stride=2, samples_per_shot=64,
                         beamforming_angles_deg=[[0,0], [30,0]])
        self.lengths = prefix_lengths(self.meta, 3)
        self.prefix = np.arange(64)[None,None,:] < self.lengths[...,None]
        rng = np.random.default_rng(23)
        self.empty = rng.normal(size=(3,2,64))
        self.raw = self.empty + np.where(self.prefix, 0., rng.normal(size=self.empty.shape))

    def test_signed_gain_and_unobservable_port_are_independent_of_late_echo(self):
        self.assertEqual(self.lengths[:, 0].tolist(), [3,3,3])
        self.empty[2][self.prefix[2]] = 0.
        observed = self.empty * np.array([1.01, .997, 1.1])[:, None,None]
        corrected, report = calibrate_pair(self.raw, observed, self.meta, 3)
        np.testing.assert_allclose(report["gain_by_receiver"], [1/1.01, 1/.997, 1.], rtol=1e-14)
        self.assertEqual(report["identifiable_by_receiver"], [True,True,False])
        np.testing.assert_allclose(corrected[:2], self.empty[:2], rtol=1e-14)
        np.testing.assert_array_equal(corrected[2], observed[2])
        altered = self.raw.copy()
        altered[~self.prefix] += 1e8
        self.assertEqual(report, calibrate_pair(altered, observed, self.meta, 3)[1])
        capped = fit_direct_gains(self.empty*3., self.empty, self.lengths)
        self.assertEqual(capped.gain.tolist(), [2.,2.,1.])
        self.assertEqual(capped.clipped.tolist(), [True,True,False])

    def test_clean_directory_preserves_bytes_and_ignores_unrelated_files(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"raw"
            source.mkdir()
            (source/"acquisition.json").write_text(json.dumps(self.meta, indent=3))
            np.save(source/"empty_sensor_samples.npy", self.empty)
            np.save(source/"sensor_samples.npy", self.raw)
            (source/"grid.npy").write_text("not an array")
            (source/"scene.json").write_text("not JSON")
            destination = root/"calibrated"
            destination.mkdir()
            report = calibrate_scene(source, destination, 3)
            self.assertEqual(report["gain_by_receiver"], [1.,1.,1.])
            self.assertEqual(report["relative_prefix_residual_by_receiver"], [0.,0.,0.])
            self.assertEqual(set(p.name for p in destination.iterdir()),
                             {"acquisition.json","sensor_samples.npy","empty_sensor_samples.npy","reference_calibration.json"})
            for name in ("sensor_samples.npy", "acquisition.json", "empty_sensor_samples.npy"):
                self.assertEqual((source/name).read_bytes(), (destination/name).read_bytes())
            with self.assertRaisesRegex(ValueError, "separate"):
                calibrate_scene(source, source, 3)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                calibrate_scene(source, root/"bad", -1)
            self.assertFalse((root/"bad").exists())

            self.meta["sensor_layout"] = "direction,antenna,timestep"
            (source/"acquisition.json").write_text(json.dumps(self.meta))
            np.save(source/"empty_sensor_samples.npy", self.empty.swapaxes(0,1)*1.01)
            np.save(source/"sensor_samples.npy", self.raw.swapaxes(0,1))
            report = calibrate_scene(source, root/"reversed", 3)
            np.testing.assert_allclose(report["gain_by_receiver"], [1/1.01]*3)
            np.testing.assert_allclose(np.load(root/"reversed"/"empty_sensor_samples.npy"), self.empty.swapaxes(0,1))
            self.assertTrue((source/"grid.npy").exists())


if __name__ == "__main__":
    unittest.main()
