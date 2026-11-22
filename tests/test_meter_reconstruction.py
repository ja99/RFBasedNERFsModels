import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import MeterReconstruction as m
import Reconstruct


def synthetic(path, count=30, start=0, sealed=False):
    rng = np.random.default_rng(17 + start)
    base = np.array([(5, y, z) for y in (12, 17, 22, 27) for z in (12, 17, 22, 27)])
    tx = base + (0, 1, 0)
    acq = dict(antenna_positions_cells=base.tolist(), transmit_positions_cells=tx.tolist(),
               grid_spacing_m=.008, time_step_s=.008/m.C, sample_stride=1, samples_per_shot=128,
               grid_shape=[49, 42, 42], pml_thickness=3, grid_origin_cells=[5, 3, 3],
               voltage_units="V", beamforming_angles_deg=[[0, 0]])
    centers = rng.uniform((18., 12., 11.), (39., 29., 30.), size=(count, 3))
    radii = rng.uniform(2.5, 4.25, size=count)
    t = np.arange(128)
    empty = np.sin(t/3.) * np.exp(-((t-5)/3.)**2) * .2
    empty = np.tile(empty, (len(tx), 1, 1)).astype("f4")
    raw = np.tile(empty, (count, 1, 1, 1))
    for i, (center, r) in enumerate(zip(centers, radii)):
        rt = np.linalg.norm((base + tx)/2 - center, axis=1)
        tt = np.linalg.norm(tx - center, axis=1).min()
        phase = t[None, :] - (rt + tt - 2*r + 3)[:, None]
        raw[i, :, 0, :] += np.sin(phase*1.2)*np.exp(-(phase/(1+r/3))**2)*r**2/(rt[:, None]**2 * tt)
    split = np.array(["test"]*count) if sealed else np.array(["validation" if i%6==0 else "train" for i in range(count)])
    sides = np.cbrt(4*np.pi/3)*radii[:, None] * [1.23, 1, 1/1.23]
    labels = dict(centers_cells=centers, radius_cells=radii, sizes_cells=sides, euler_deg=rng.uniform(-180,180,(count,3)),
                  kind=np.array(["sphere" if i%2 else "box" for i in range(count)]))
    np.savez(path, sensor_samples=raw, empty_sensor_samples=empty, acquisition_json=json.dumps(acq),
             scene_ids=[f"p{i+start}" for i in range(count)], split=split, **labels)
    return acq, labels


class TestMeterReconstruction(unittest.TestCase):
    def test_arrivals_localize_with_shared_tx_excitation_and_distinct_rx_paths(self):
        with TemporaryDirectory() as td:
            acq, _ = synthetic(Path(td)/"shot.npz", count=2)
        sensor = m.TimingGeometry(acq)
        p = np.array([36.4, 27.8, 11.4])
        times = (np.linalg.norm(p-sensor.tx, axis=1).min() + np.linalg.norm(p-sensor.rx, axis=1))/sensor.scale
        estimated, residual = sensor.locate(times)
        np.testing.assert_allclose(p, estimated, atol=.0001)
        self.assertLess(residual, .0001)

    def test_roi_masks_reproduce_full_lattice_rotated_truth(self):
        with TemporaryDirectory() as td:
            acq, labels = synthetic(Path(td)/"shot.npz", count=2)
        origin, end = m.roi(acq)
        grid = np.indices(end-origin).transpose(1,2,3,0) + origin
        box = np.abs((grid - labels["centers_cells"][0]) @ Rotation.from_euler("xyz", labels["euler_deg"][0], degrees=True).as_matrix())
        inside = np.all(box < labels["sizes_cells"][0]/2, axis=-1)
        np.testing.assert_array_equal(m.truth_voxels(labels, 0, acq), np.flatnonzero(inside))
        sphere = m.shape_vectors(labels)[1]
        np.testing.assert_array_equal(m.predicted_voxels(sphere, acq), m.truth_voxels(labels, 1, acq))

    def test_fit_predict_and_evaluate_are_blind_and_reproducible(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            train_path, sealed, unlabeled, ckpt = (root/name for name in ["train.npz", "sealed.npz", "raw.npz", "model.npz"])
            synthetic(train_path)
            acq, _ = synthetic(sealed, count=5, start=100, sealed=True)
            with np.load(sealed) as data:
                raw = {k:data[k] for k in ("acquisition_json","sensor_samples","empty_sensor_samples","scene_ids")}
            np.savez(unlabeled, **raw)
            validation = m.train(train_path, ckpt)
            self.assertGreater(validation["mean_scene_iou"], .25)
            calls = []
            predict, truth = m.Estimator.predict, m.geometry

            def record_predict(*args, **kwargs):
                calls.append("predict")
                return predict(*args, **kwargs)

            def record_truth(*args, **kwargs):
                calls.append("labels")
                return truth(*args, **kwargs)

            with patch.object(m.Estimator, "predict", record_predict), patch.object(m, "geometry", record_truth):
                report = m.evaluate(sealed, ckpt)
            self.assertEqual(calls, ["predict", "labels"])
            scores = report["results"]
            self.assertGreater(scores["reconstruction"]["summary"]["mean_scene_iou"], .25)
            self.assertGreater(scores["reconstruction"]["summary"]["mean_scene_iou"], scores["shuffled_echo"]["summary"]["mean_scene_iou"])
            m.predict_scene(unlabeled, ckpt, "p100", root/"out.npz")
            with np.load(root/"out.npz") as result:
                self.assertEqual(result["occupancy"].sum(), scores["reconstruction"]["scenes"][0]["occupied_predicted"])
            with patch.object(Reconstruct, "release_asset", return_value=ckpt) as asset:
                Reconstruct.reconstruct(unlabeled, root/"wrapper.npz", mode="metre", scene_id="p100")
                self.assertEqual(asset.call_args.args[0], "metre-single-shot.npz")
                with np.load(root/"out.npz") as direct, np.load(root/"wrapper.npz") as wrapped:
                    self.assertEqual(set(direct.files), set(wrapped.files))
                    for key in direct.files:
                        np.testing.assert_array_equal(direct[key], wrapped[key])
                Reconstruct.reconstruct(unlabeled, root/"calibrated.npz", mode="metre", scene_id="p100", clearance_cells=3)
                with np.load(root/"calibrated.npz") as calibrated:
                    self.assertEqual(json.loads(calibrated["reference_calibration_json"].item())["gain_ports"], 16)
                with self.assertRaisesRegex(ValueError, "Select --scene-id"):
                    Reconstruct.reconstruct(unlabeled, root/"wrapper.npz", mode="metre")
            silent = dict(sensor_samples=raw["empty_sensor_samples"][None], empty_sensor_samples=raw["empty_sensor_samples"],
                          acquisition_json=raw["acquisition_json"])
            np.savez(root/"silent.npz", **silent)
            m.predict_scene(root/"silent.npz", ckpt, None, root/"silent-out.npz")
            with np.load(root/"silent-out.npz") as result:
                self.assertFalse(result["occupancy"].any())
                self.assertFalse(result["nonzero_echo"].item())
            with self.assertRaisesRegex(ValueError, "independent test"):
                m.evaluate(train_path, ckpt)
            raw["acquisition_json"] = json.dumps({**acq, "sample_stride": 2})
            np.savez(unlabeled, **raw)
            with self.assertRaisesRegex(ValueError, "differs"):
                m.predict_scene(unlabeled, ckpt, "p100", root/"out.npz")


if __name__ == "__main__":
    unittest.main()
