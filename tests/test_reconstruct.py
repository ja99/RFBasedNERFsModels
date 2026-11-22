import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from EchoKernel import DIRECTIONS, EchoKernel, _echo_features, _support
from EchoMaterial import EchoMaterial, _features
import Reconstruct


class ReconstructTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scene, self.weights = self.root / "raw", self.root / "weights"
        self.scene.mkdir()
        self.weights.mkdir()
        self.output = self.root / "prediction.npz"
        self.manifest = {"release": Reconstruct.RELEASE, "assets": {}}
        self.manifest_path = self.root / "manifest.json"
        self.addCleanup(patch.stopall)
        patch.object(Reconstruct, "MANIFEST", self.manifest_path).start()
        self.open_url = patch.object(Reconstruct, "urlopen", side_effect=AssertionError("Unexpected download")).start()
        self.shape = (8, 9, 10)
        self.mask = np.zeros(self.shape, dtype=bool)
        self.mask[2:6, 3:7, 1:4] = True
        self.wave = np.arange(64).reshape(2, 1, 32).astype("f8") / 64

    def register(self, filename):
        payload = (self.weights / filename).read_bytes()
        self.manifest["assets"][filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        self.manifest_path.write_text(json.dumps(self.manifest))

    def raw_pair(self, protocol):
        metadata = {"sensor_layout": "direction,antenna,timestep", "grid_origin_cells": [7, 4, 4],
                    "grid_spacing_m": Reconstruct.SPACING_M[protocol]}
        (self.scene / "acquisition.json").write_text(json.dumps(metadata))
        reference = np.full((1, 2, 32), 1e10)
        np.save(self.scene / "empty_sensor_samples.npy", reference)
        np.save(self.scene / "sensor_samples.npy", reference + self.wave.swapaxes(0, 1))
        return metadata

    def single_assets(self, protocol, metadata, material=True):
        settings = dict(acquisition=metadata, sensor_shape=list(self.wave.shape), grid_shape=list(self.shape),
                        sensor_offset=.25, sensor_scale=.5)
        choice = dict(method="ridge", mix=[1/3]*3, width=1., offset=0.)
        kernel_meta = dict(format_version=1, settings=settings, choice=choice)

        def transforms(groups):
            arrays = {}
            for i, group in enumerate(groups):
                arrays[f"mean_{i}"] = np.zeros(group.size, dtype="f4")
                arrays[f"scale_{i}"] = np.ones(group.size, dtype="f4")
                arrays[f"basis_{i}"] = np.ones((group.size, 1), dtype="f4")
            return arrays

        kernel = f"{protocol}-kernel.npz"
        np.savez(self.weights / kernel, meta=json.dumps(kernel_meta), **transforms(_echo_features(self.wave)),
                 anchors=np.zeros((1, 3)), design_mean=np.zeros(4), coef=np.zeros((4, len(DIRECTIONS))),
                 target_mean=_support(self.mask))
        self.register(kernel)
        if material:
            settings = dict(settings, sensor_scale=.125)
            mean = [4., .08] if protocol == "dielectric" else [56., 10.]
            material_meta = dict(format_version=1, model_type="echo_material", settings=settings, background=[1., 0.],
                                 train_mean=mean, choices=[dict(method="train_mean")]*2)
            filename = f"{protocol}-material.npz"
            np.savez(self.weights / filename, meta=json.dumps(material_meta),
                     **transforms(_features(self.wave, metadata)))
            self.register(filename)
            return mean

    def test_single_routes_by_acquisition_and_applies_both_training_normalizers(self):
        for protocol in ("dielectric", "conductive"):
            with self.subTest(protocol=protocol):
                metadata = self.raw_pair(protocol)
                mean = self.single_assets(protocol, metadata)
                with patch.object(EchoKernel, "predict", autospec=True, side_effect=EchoKernel.predict) as kernel, \
                     patch.object(EchoMaterial, "predict", autospec=True, side_effect=EchoMaterial.predict) as material:
                    Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights)
                np.testing.assert_array_equal(kernel.call_args.args[1], (self.wave - .25) / .5)
                np.testing.assert_array_equal(material.call_args.args[1], (self.wave - .25) / .125)
                with np.load(self.output, allow_pickle=False) as result:
                    np.testing.assert_array_equal(result["occupied"], self.mask)
                    np.testing.assert_array_equal(result["sdf_cells"] < 0, self.mask)
                    np.testing.assert_allclose(result["permittivity"], np.where(self.mask, mean[0], 1.))
                    np.testing.assert_allclose(result["conductivity_s_per_m"], np.where(self.mask, mean[1], 0.))
                    np.testing.assert_array_equal(result["grid_origin_cells"], [7, 4, 4])
                    self.assertEqual(float(result["grid_spacing_m"]), metadata["grid_spacing_m"])
                previous = self.output.read_bytes()
                (self.scene / "acquisition.json").write_text(json.dumps(dict(metadata, samples_per_shot=8)))
                with self.assertRaisesRegex(ValueError, "acquisition"):
                    Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights)
                self.assertEqual(self.output.read_bytes(), previous)

    def test_mixed_uses_volume_checkpoint_and_outputs_geometry(self):
        import torch
        from EchoReconstructor import VolumeReconstructor, trace_features

        metadata = self.raw_pair("dielectric")
        settings = dict(acquisition=metadata, sensor_shape=list(self.wave.shape), sensor_storage="antenna_direction,time")
        protocol = dict(settings=settings, normalization={"sensor_offset": .2, "sensor_scale": .5})
        feature_shape = trace_features(torch.from_numpy(self.wave[None].astype("f4"))).shape[1:]
        model = VolumeReconstructor(feature_shape, self.shape, hidden=16)
        model.spatial_mean.fill_(1.)
        model.spatial_mean[0] = -1.
        torch.save(dict(architecture=model.architecture, config=model.config, weights=model.state_dict(),
                        waveform_mode="calibrated", preprocessing_id="example", training_protocol=protocol),
                   self.weights / "mixed-volume.pt")
        self.register("mixed-volume.pt")
        Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, mode="mixed")
        with np.load(self.output, allow_pickle=False) as result:
            self.assertEqual(result["sdf_cells"].shape, self.shape)
            np.testing.assert_array_equal(result["sdf_cells"] < 0, result["occupied"])
            self.assertGreater(result["occupied"].sum(), 0)
            self.assertLess(result["occupied"].sum(), self.mask.size)
            self.assertEqual(set(result.files), {"sdf_cells", "occupied", "scene_id", "grid_origin_cells", "grid_spacing_m"})
        (self.scene / "acquisition.json").write_text(json.dumps(dict(metadata, frequency_hz=1.)))
        with self.assertRaisesRegex(ValueError, "acquisition"):
            Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, mode="mixed")
        self.raw_pair("conductive")
        with self.assertRaisesRegex(ValueError, "dielectric"):
            Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, mode="mixed")

    def test_refine_isolates_sibling_imports_and_publishes_simulator_result(self):
        metadata = self.raw_pair("conductive")
        self.single_assets("conductive", metadata, material=False)
        sibling = self.root / "sibling checkout"
        sibling.mkdir()
        (sibling / "helpers.py").write_text("fixture_marker = 57\n")
        (sibling / "RefinePhantom.py").write_text(
            "import sys\nfrom pathlib import Path\nimport numpy as np\nfrom helpers import fixture_marker\n"
            "scene, initial, out, *options = sys.argv[1:]\n"
            "assert options == ['--material', 'thesis', '--device', 'cpu']\n"
            "assert len(list(Path(scene).iterdir())) == 3\n"
            "sdf = np.load(initial, allow_pickle=False)\n"
            "np.savez(out, sdf_cells=sdf, occupied=sdf < 0, permittivity=np.where(sdf < 0, fixture_marker, 1))\n")
        Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, simulator=sibling, refine=True)
        with np.load(self.output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["occupied"], self.mask)
            np.testing.assert_array_equal(result["permittivity"], np.where(self.mask, 57, 1))
        with self.assertRaisesRegex(ValueError, "compact"):
            Reconstruct.reconstruct(self.scene, self.output, mode="metre", refine=True)

    def test_mixed_refinement_passes_rf_selected_slots_and_corrected_recordings_to_sibling(self):
        import torch
        from EchoSet import EchoSet, SetPredictor

        metadata = self.raw_pair("dielectric")
        metadata.update(sensor_layout="antenna,direction,timestep", antenna_positions_cells=[[7,6,6],[7,9,6]],
                        transmit_positions_cells=[[7,7,6],[7,10,6]], beamforming_angles_deg=[[0,0]],
                        time_step_s=8.58e-12, sample_stride=2, samples_per_shot=32,
                        grid_shape=[19, 17, 18], pml_thickness=4)
        (self.scene / "acquisition.json").write_text(json.dumps(metadata))
        self.wave[..., :3] = 0.
        reference = np.random.default_rng(18).normal(size=self.wave.shape)
        drifted = reference * np.array([1.02, .98])[:, None, None]
        np.save(self.scene / "sensor_samples.npy", reference + self.wave)
        np.save(self.scene / "empty_sensor_samples.npy", drifted)

        settings = dict(acquisition=metadata, sensor_shape=list(self.wave.shape), grid_shape=list(self.shape),
                        sensor_offset=.25, sensor_scale=.5)
        transforms = [[torch.zeros(g.size), torch.ones(g.size), torch.ones((g.size, 1))]
                      for g in _echo_features(self.wave)]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            model = EchoSet(3, self.shape, hidden=16, shape_rank=2, landmarks=3)
        checkpoint = self.weights / "mixed-set.pt"
        torch.save(dict(format_version=1, model_type="echo_set", config=model.config, weights=model.state_dict(),
                        transforms=transforms, settings=settings, prior=torch.from_numpy(self.mask), prior_id="toy",
                        training_ids=[], validation={"offset": 0., "threshold": 0.}), checkpoint)
        self.register(checkpoint.name)
        normalized = ((self.wave - .25) / .5).astype("f4")
        _, masks, centers, p_two = SetPredictor(checkpoint).fields(normalized[None])[0]
        self.assertEqual(len(masks), 2)
        sibling = self.root / "simulator"
        sibling.mkdir()
        (sibling / "RefineInstances.py").write_text(
            "import sys\nfrom pathlib import Path\nimport numpy as np\n"
            "scene, slots, out = map(Path, sys.argv[1:4])\n"
            "assert sys.argv[4:] == ['--material', 'dielectric', '--device', 'cpu']\n"
            "assert not (scene/'grid.npy').exists() and not (scene/'scene.json').exists()\n"
            "residual = np.load(scene/'sensor_samples.npy') - np.load(scene/'empty_sensor_samples.npy')\n"
            "with np.load(slots, allow_pickle=False) as inputs:\n"
            "    np.savez(out, instance_masks=inputs['masks'], centers=inputs['centers_cells'],\n"
            "             probability=inputs['probability_two'], used_residual=residual)\n")
        Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, simulator=sibling,
                                mode="mixed", refine=True, clearance_cells=3)
        with np.load(self.output, allow_pickle=False) as result:
            np.testing.assert_array_equal(result["instance_masks"], masks)
            np.testing.assert_allclose(result["centers"], centers, atol=1e-6)
            self.assertAlmostEqual(float(result["probability"]), p_two, places=6)
            np.testing.assert_allclose(result["used_residual"], self.wave, atol=1e-14)
            gains = json.loads(result["reference_calibration_json"].item())["gain_by_receiver"]
            np.testing.assert_allclose(gains, [1/1.02, 1/.98])
        np.testing.assert_array_equal(np.load(self.scene / "empty_sensor_samples.npy"), drifted)
        (self.scene / "acquisition.json").write_text(json.dumps(dict(metadata, frequency_hz=1.)))
        with self.assertRaisesRegex(ValueError, "acquisition"):
            Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, simulator=sibling,
                                    mode="mixed", refine=True)

    def test_gain_corrected_pair_reaches_single_and_refined_paths(self):
        metadata = self.raw_pair("dielectric")
        metadata.update(antenna_positions_cells=[[7,6,6],[7,9,6]], transmit_positions_cells=[[7,7,6],[7,10,6]],
                        beamforming_angles_deg=[[0,0]], time_step_s=8.58e-12, sample_stride=2, samples_per_shot=32,
                        grid_shape=[19, 17, 18], pml_thickness=4)
        (self.scene / "acquisition.json").write_text(json.dumps(metadata))
        clean = np.random.default_rng(17).normal(size=(1,2,32))
        self.wave[..., :3] = 0.
        measured = clean + self.wave.swapaxes(0, 1)
        drift = clean * np.array([1.01, .99])[None,:,None]
        np.save(self.scene / "sensor_samples.npy", measured)
        np.save(self.scene / "empty_sensor_samples.npy", drift)
        self.single_assets("dielectric", metadata)
        with patch.object(EchoKernel, "predict", autospec=True, side_effect=EchoKernel.predict) as kernel:
            Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, clearance_cells=3)
        np.testing.assert_allclose(kernel.call_args.args[1], (self.wave-.25)/.5, atol=1e-7)
        with np.load(self.output) as prediction:
            gains = json.loads(prediction["reference_calibration_json"].item())["gain_by_receiver"]
            np.testing.assert_allclose(gains, [1/1.01, 1/.99])
            np.testing.assert_array_equal(prediction["occupied"], self.mask)
        simulator = self.root / "RefinePhantom.py"
        simulator.write_text("from pathlib import Path\nimport sys, numpy as np\n"
                             "scene, initial, output = map(Path, sys.argv[1:4])\n"
                             "measured = np.load(scene/'sensor_samples.npy')\n"
                             "reference = np.load(scene/'empty_sensor_samples.npy')\n"
                             "np.savez(output, used_residual=measured-reference)\n")
        Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights, clearance_cells=3,
                                simulator=simulator, refine=True)
        with np.load(self.output) as out:
            self.assertIn("reference_calibration_json", out.files)
            np.testing.assert_allclose(out["used_residual"], self.wave.swapaxes(0,1), atol=1e-14)
        np.testing.assert_array_equal(np.load(self.scene/"empty_sensor_samples.npy"), drift)

    def test_cache_verifies_existing_downloads_and_preserves_prior_file_on_failure(self):
        name = "dielectric-kernel.npz"
        path = self.weights / name
        good, faulty = b"verified release fixture", b"broken!! release fixture"
        self.assertEqual(len(good), len(faulty))
        path.write_bytes(good)
        self.register(name)
        self.assertEqual(Reconstruct.release_asset(name, self.weights, self.manifest), path)
        self.open_url.assert_not_called()
        path.write_bytes(faulty)
        self.open_url.side_effect = lambda *args, **kwargs: io.BytesIO(faulty)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            Reconstruct.release_asset(name, self.weights, self.manifest)
        self.assertEqual(path.read_bytes(), faulty)
        self.assertEqual(list(self.weights.iterdir()), [path])
        self.open_url.side_effect = lambda *args, **kwargs: io.BytesIO(good)
        Reconstruct.release_asset(name, self.weights, self.manifest)
        self.assertEqual(path.read_bytes(), good)
        self.assertEqual(self.open_url.call_args.args, (f"{Reconstruct.RELEASE_URL}/{name}",))
        path.unlink()
        Reconstruct.release_asset(name, self.weights, self.manifest)
        self.assertEqual(path.read_bytes(), good)
        (self.scene / "acquisition.json").write_text(json.dumps({"grid_spacing_m": .25}))
        with self.assertRaisesRegex(ValueError, "grid_spacing_m"):
            Reconstruct.reconstruct(self.scene, self.output, weights_dir=self.weights)


if __name__ == "__main__":
    unittest.main()
