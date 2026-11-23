# RF scene reconstruction

The simulator writes five steered shots and a matched empty-scene recording. The models take the signed difference between those recordings and predict fields on the voxel grid. The commands below assume the simulator and models are sibling checkouts and Python 3.12 is active.

## Run a trained reconstruction

The example contains two RF recordings and the array calibration. It is the second previously unseen scene in the low-loss test set. From this directory, run the following commands with Python 3.12.

```bash
python -m pip install -r requirements.txt
python Reconstruct.py examples/dielectric reconstruction.npz --device cpu
python -c "import numpy as np; a=np.load('reconstruction.npz'); print(a.files, a['occupied'].sum())"
```

The first run fetches the signed-distance and material model weights from the release and verifies their SHA-256 digests. Use `--weights-dir /workspace/rf-models` to keep a local copy on a Runpod volume. The output contains the signed distance in grid cells, occupancy, relative permittivity, conductivity in S/m, cell size in metres and grid origin. Negative distances are inside the predicted object. [Trained weights](https://github.com/ja99/RFBasedNERFsModels/releases/tag/phantom-benchmark-v1)

For one homogeneous sphere or rotated box, `--refine` fits both candidate shapes and their material against the measured voltages. Install the sibling simulator requirements, then run the following command on a CUDA machine with `nvcc` installed. The CPU solver also supports refinement with a C++17/OpenMP compiler.

```bash
python -m pip install -r ../RFBasedNERFsSimulator/requirements.txt
python Reconstruct.py examples/dielectric refined.npz --refine --device cuda
```

For a low-loss scene containing one or two separated objects, `--mode mixed` runs a volume decoder trained on both counts. It exports geometry. Use `--mode mixed --refine` to initialize up to two individual masks from RF and fit their joint waveform. The output includes reconstructed material fields and individual masks. The standalone two-slot estimator also exposes its masks through `EchoSet.py infer --instances`. Each model checks that the supplied array, pulse, sample timing and empty reference use its fitted calibration. For the metre-scale, single-excitation protocol, use its separate arrival-time estimator. The metre example is the second sealed measurement, a rotated box reconstructed at IoU 0.585 and center error 1.28 cm. The first measurement is preserved in `examples/metre-miss`. Its 23.46 cm center error produced zero overlap. Both examples contain only measured and empty waveforms, IDs and calibration. The estimator exports the predicted occupied volume and estimated center on the 8 mm lattice.

```bash
python Reconstruct.py examples/metre metre-reconstruction.npz --mode metre
```

New measurements use a directory containing `sensor_samples.npy`, `empty_sensor_samples.npy` and `acquisition.json`. For the compact protocol the arrays are ordered `[16 receivers, 5 shots, 128 samples]`. The two recordings must share the same acquisition settings. The example runs without any voxel labels. [Experiment design and scores in the private thesis repository](https://github.com/ja99/MasterThesisLatex/blob/main/diagnostics/benchmark_results.md)

When a laboratory setup provides an independently known empty forward region, `--clearance-cells N` fits the reference gain independently for each receiver using only samples that precede the earliest possible object return. The compact benchmark guarantees two cells of clearance. The default uses the supplied empty recording directly. To export a corrected reference and its diagnostics, run `python CalibrateReference.py INPUT CALIBRATED --clearance-cells N`.

## Reproduce the small CPU experiment

Run these commands from the models directory. Choose a new data directory for each simulation. On a pod, set `RF_RUN` to a directory on its mounted volume to keep simulation data, the archive and checkpoints together.

```bash
python -m pip install -r ../RFBasedNERFsSimulator/requirements.txt -r requirements.txt
export RF_RUN="$PWD/rf-run"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
rf_acquisition=(--seed 211 --shape 32 32 32 --pml 4 --standoff-cells 3
                --pitch-cells 3 --samples 128 --jobs 2 --backend torch)
python ../RFBasedNERFsSimulator/Acquisition.py "$RF_RUN/scenes" --scenes 8 "${rf_acquisition[@]}"
python SaveAsHDF5.py "$RF_RUN/scenes" --seed 42
python FullyINR.py train --data "$RF_RUN/scenes" --compact --max-epochs 100 \
    --seed 42 --output-dir "$RF_RUN/models"
checkpoint=$(find "$RF_RUN/models" -name best.ckpt -print -quit)
python FullyINR.py evaluate --data "$RF_RUN/scenes" --checkpoint "$checkpoint"
python ../RFBasedNERFsSimulator/Acquisition.py "$RF_RUN/new-scenes" --start-index 8 --scenes 4 "${rf_acquisition[@]}"
python SaveAsHDF5.py "$RF_RUN/new-scenes" --normalization-from "$RF_RUN/scenes"
python FullyINR.py evaluate --data "$RF_RUN/new-scenes" --checkpoint "$checkpoint"
python TemplateBaseline.py "$RF_RUN/scenes" "$RF_RUN/scenes" "$RF_RUN/new-scenes"
```

Training prints the checkpoint path selected by validation loss. `evaluate` scores every voxel in every test scene. `render` with the same arguments opens the first test scene. Reuse the specific checkpoint printed by the desired run if the output directory contains several fits.

`TemplateBaseline.py` selects the most similar training echo by cosine similarity and returns that training scene's SDF. It gives a measurement-dependent control for the neural network and can only reproduce shapes already present in the training set. To score both the original and newly acquired held-out scenes, run `python TemplateBaseline.py TRAIN_DIR TRAIN_DIR NEW_DIR`.

The 32³ experiment is a small near-field integration check. Both aperture pitch and sensor stand-off are deliberately short. With the acquisition defaults, a 4×4 transversely polarized array uses a half-wavelength pitch at 2.4 GHz. The default cell spacing is about 1.179 mm, the field grid is 128×216×216 cells including 15 cells of PML per face, and each shot stores 750 time samples. Both the physical domain and random object size differ from the original metre-scale experiment. Quantitative physical comparisons require mesh and PML convergence on a fixed physical scene.

For a study at the thesis's physical object sizes, `--object-unit-m 0.00625` keeps the original 5–24-unit sphere radii and 5–49-unit box side lengths in physical units as the mesh changes. Choose `--shape` and `--pml` in cells for the intended physical domain. For the same individual scene on two meshes, pass explicit `WorldObject` instances to `Acquisition.write_scene`, dividing their centers and physical dimensions by each mesh's spacing. The receiver integrates the electric field over an idealized one-cell element. A physical convergence experiment also needs a consistent physical source moment and receiver length, or a consistently normalized scattering observable.

## Result of the compact experiment

All six training scenes and five held-out scenes contained single spheres. The validation scene contained a box. On the full held-out volumes, closest-echo template matching scored **0.540 mean occupancy IoU**. Choosing uniformly among the same six training shapes scored **0.206** averaged over all 30 test/template pairs. The compact neural model averaged **0.020** over training seeds 7, 23 and 42 (`0.000`, `0.010`, `0.049`). The coordinate-only mean of the six training distance fields scored **0.093**. The calibrated echoes helped select a similar sphere. Generalization to new shape families and the metre-scale domain remains open.

## Data and evaluation

To fit the compact geometry estimator using coherent I/Q, run `python EchoKernel.py fit --data TRAIN_ARCHIVE --output iq.npz --representation iq`. For a matched comparison, pass the same archive and reflection settings to each fit. The default `voltage` uses signed passband samples. The `iq` and `magnitude` modes share envelope-based energy and arrival-time features. The former pools both quadratures and the latter pools their magnitude. The `voltage-envelope-stats` control combines the original pooled-voltage features with the same envelope statistics. All modes use the complete recorded bandwidth before pooling. The checkpoint records the representation for evaluation and raw-scene inference.

- Each scene directory contains `grid.npy` with channels relative permittivity, conductivity in S/m, and signed distance in cells. In newly generated scenes the distance measures the boundary of the occupied voxel union. An occupied cell has negative distance. The near and far faces of a single voxel are each half a cell from its center.
- `sensor_samples.npy` and `empty_sensor_samples.npy` have axes `[antenna, direction, timestep]` and units of volts. `acquisition.json` records grid origin, spacing, antenna positions, beam angles, simulation time step and sample stride. Each shot starts with fresh electromagnetic and PML fields.
- The automatic recording window covers transmission and a free-space round trip to the farthest grid corner. Set `--samples` and `--sample-stride` together to study later multipath or slower propagation through material. The recorded span is (samples − 1) × stride × simulation time step. Resume acquisition in a partially completed folder using `--start-index` for the next scene and the original seed.
- The converter pairs files by scene ID. Identical material grids stay in the same split, including copies with different names. In the eight-scene example six distinct scenes train the model, one selects the checkpoint, and one measures held-out performance.
- Training scenes set the per-channel grid scales and maximum absolute echo. Held-out values may exceed the training range. The archive and checkpoint retain the transform and split so evaluation uses the same physical SDF zero and input calibration. Evaluation reports per-scene and pooled occupancy intersection-over-union (IoU), occupied voxel counts, and mean absolute SDF error in grid cells.
- To evaluate a newly acquired folder of scenes, use the same sensor and grid configuration, distinct scene IDs and fresh geometry. Run `python SaveAsHDF5.py NEW_SCENES --normalization-from "$RF_RUN/scenes/dataset.hdf5"`, then run `evaluate --data NEW_SCENES` with the trained checkpoint. All scenes in that new archive enter its test split.

## Historical data

An old `dataset.hdf5` can still be read in place. Its original preprocessing and physical SDF zero were never stored. Legacy rendering uses matched normalized SDF slices and reports normalized error. A deterministic fallback split is available, but the original training membership cannot be recovered from that archive.

The field solver, waveform steering and voxel-boundary fixes take effect during acquisition. Regenerate scenes and train fresh weights to use these corrections. Converting historical raw files preserves the voltages and labels they actually contain.

For historical raw scene folders, select `--raw` explicitly or supply an empty recording acquired with the same settings via `--empty-reference`. A shared reference for a scene with recorded metadata needs matching `acquisition.json` alongside the reference. Declare single-shot arrays with `--single-shot`, and verified antenna-major five-shot packed arrays with `--sensor-layout packed-antenna-direction`. A direction-first 3-D array uses `--sensor-layout direction,antenna,timestep`. Use `--crop-x-end 15` only to reproduce the additional old x crop. Converting writes a new `dataset.hdf5` in that directory, so keep the historical archive separately if it is needed for an old checkpoint.

Historical checkpoints need their original raw archive. For a FullyINR checkpoint missing its architecture metadata, loading recovers layer widths from the weights and assumes the thesis training script's GELU activation. Supply `--activation Tanh` at evaluation or rendering if that run used Tanh. New checkpoints trained on empty-subtracted signals require compatible calibrated data. The historical voxel-sampling and RF probes remain reproducible against the archived Git revisions in the [audit notebook in the private thesis repository](https://github.com/ja99/MasterThesisLatex/blob/main/diagnostics/rf_reconstruction_audit.ipynb).
