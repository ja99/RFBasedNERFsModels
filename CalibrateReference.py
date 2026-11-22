"""Fit an empty recording to the direct path in a paired laboratory acquisition.

The caller supplies the known minimum forward x offset, in grid cells from the array plane, of
any material that differs between the empty and measured scene. The prefix also assumes the two
recordings have the same clock, transmit schedule and receiver layout. One bounded multiplicative
gain is fitted per receiver across its shots. Unidentifiable receivers retain their input reference.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

CLEARANCE_HELP = ("Known minimum forward distance in cells from the array plane to any changed "
                  "material, across the whole aperture and field of view")
GAIN_LIMITS = (.5, 2.)


def prefix_lengths(acquisition, clearance):
    """Number of recorded samples strictly before the first possible scattered Yee E update."""
    required = {"antenna_positions_cells", "transmit_positions_cells", "grid_origin_cells", "grid_shape",
                "pml_thickness", "beamforming_angles_deg", "grid_spacing_m", "time_step_s", "sample_stride",
                "samples_per_shot"}
    missing = sorted(required.difference(acquisition))
    if missing:
        raise ValueError("Direct-path gain calibration needs acquisition fields " + ", ".join(missing))
    rx = np.asarray(acquisition["antenna_positions_cells"])
    tx = np.asarray(acquisition["transmit_positions_cells"])
    if not isinstance(clearance, (int, np.integer)) or clearance < 0:
        raise ValueError("The known clearance must be a nonnegative integer number of cells")
    if rx.ndim != 2 or rx.shape[1:] != (3,) or not len(rx) or rx.shape != tx.shape:
        raise ValueError("Expected matching transmitter and receiver positions in grid cells")
    if not np.isfinite(rx).all() or not np.isfinite(tx).all() or np.any(tx != np.rint(tx)) or np.any(rx != np.rint(rx)):
        raise ValueError("The discrete causal bound requires on-grid receiver and transmitter coordinates")
    tx, rx = tx.astype(int), rx.astype(int)
    if np.any(rx[:, 0] != tx[0, 0]) or np.ptp(tx[:, 0]) or np.max(np.abs(tx-rx)) > 1:
        raise ValueError("Expected planar transmitters and one-cell receiver integrals from base to tip")
    origin = np.asarray(acquisition["grid_origin_cells"], dtype=int)
    end = np.asarray(acquisition["grid_shape"], dtype=int) - acquisition["pml_thickness"]
    if origin.shape != (3,) or end.shape != (3,) or np.any(end <= origin):
        raise ValueError("Invalid acquisition region of interest")
    angles = np.deg2rad(np.asarray(acquisition["beamforming_angles_deg"], dtype=float))
    if angles.ndim != 2 or angles.shape[1:] != (2,) or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Expected finite azimuth and elevation for every transmitted shot")
    dx, dt = acquisition["grid_spacing_m"], acquisition["time_step_s"]
    stride, samples = acquisition["sample_stride"], acquisition["samples_per_shot"]
    if not all(np.isfinite(v) and v > 0 for v in (dx, dt)) or not all(
            isinstance(v, (int, np.integer)) and v > 0 for v in (stride, samples)):
        raise ValueError("Expected positive cell spacing, timestep, sample stride and capture length")

    ys, zs = np.meshgrid(np.arange(origin[1], end[1]), np.arange(origin[2], end[2]), indexing="ij")
    plane = np.stack((ys.ravel(), zs.ravel()), axis=1)
    # Each staggered Yee E-to-E step advances by at most one cell per axis. Minimize the
    # TX and RX legs jointly through a possible first scatter point beyond the empty slab.
    tx_leg = np.maximum(clearance, np.abs(tx[:, None, 1:] - plane).max(-1))
    rx_leg = np.minimum(tx_leg, np.maximum(clearance, np.abs(rx[:, None, 1:] - plane).max(-1)))
    path = (rx_leg[:, None] + tx_leg[None]).min(-1)
    az, el = angles.T
    directions = np.stack((np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)), axis=1)
    times = directions @ tx.T * (dx / 299792458.)
    # Round down to allow the first source contribution to occur earlier than its actual onset.
    start = np.floor(np.maximum(times-times.min(1, keepdims=True), 0.) / dt).astype(int)
    first_possible_echo = (path[:, None, :] + start[None]).min(-1)
    return np.minimum((first_possible_echo + stride - 1) // stride, samples)


@dataclass
class GainFit:
    gain: np.ndarray
    identifiable: np.ndarray
    clipped: np.ndarray
    reference_norm: np.ndarray
    relative_prefix_residual: list
    minimum_norm: float

    def correct(self, reference):
        return self.gain[:, None, None] * reference

    def diagnostics(self):
        return dict(total_ports=len(self.gain), gain_ports=int(self.identifiable.sum()),
                    gain_by_receiver=self.gain.tolist(), identifiable_by_receiver=self.identifiable.tolist(),
                    clipped_by_receiver=self.clipped.tolist(), gain_limits=list(GAIN_LIMITS),
                    prefix_reference_l2=self.reference_norm.tolist(), minimum_prefix_reference_l2=self.minimum_norm,
                    relative_prefix_residual_by_receiver=self.relative_prefix_residual)


def fit_direct_gains(measured, reference, lengths):
    """Fit y = gain * reference using only finite prefix samples in [receiver, shot, time]."""
    measured = np.asarray(measured, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    lengths = np.asarray(lengths)
    if reference.ndim != 3 or reference.shape != measured.shape or not all(reference.shape):
        raise ValueError("Expected matching nonempty measured and empty arrays in [receiver,shot,time]")
    if (lengths.shape != reference.shape[:2] or not np.issubdtype(lengths.dtype, np.integer)
            or np.any(lengths < 0) or np.any(lengths > reference.shape[-1])):
        raise ValueError("Invalid causal prefix lengths for these recordings")
    valid = np.arange(reference.shape[-1]) < lengths[..., None]
    n_rx = len(reference)
    norms = np.linalg.norm(np.where(valid, reference, 0.).reshape(n_rx, -1), axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Nonfinite empty prefix")
    minimum = max(float(norms.max()) * 1e-7, 1e-30)
    identifiable = norms > minimum
    gains, clipped = np.ones(n_rx), np.zeros(n_rx, dtype=bool)
    residual = [None] * n_rx
    for i in np.flatnonzero(identifiable):
        r, y = reference[i][valid[i]], measured[i][valid[i]]
        if not np.isfinite(y).all():
            raise ValueError(f"Nonfinite measured prefix at receiver {i}")
        unconstrained = np.dot(r, y) / np.dot(r, r)
        gains[i] = np.clip(unconstrained, *GAIN_LIMITS)
        clipped[i] = gains[i] != unconstrained
        residual[i] = float(np.linalg.norm(y - gains[i]*r) / norms[i])
    return GainFit(gains, identifiable, clipped, norms, residual, minimum)


def calibrate_pair(measured, reference, acquisition, clearance_cells):
    """Apply causal gain calibration to a canonical [receiver, shot, time] pair."""
    lengths = prefix_lengths(acquisition, clearance_cells)
    if measured.shape != (*lengths.shape, acquisition["samples_per_shot"]) or reference.shape != measured.shape:
        raise ValueError("Both recordings must match the port, shot and sample counts in acquisition.json")
    if not np.isfinite(measured).all() or not np.isfinite(reference).all():
        raise ValueError("The measured and empty recordings must be finite")
    fit = fit_direct_gains(measured, reference, lengths)
    report = dict(method="causal_direct_path_gain", minimum_material_x_offset_cells=int(clearance_cells),
                  prefix_samples_by_receiver_and_shot=lengths.tolist(), **fit.diagnostics())
    return fit.correct(reference), report


def calibrate_scene(source, destination, clearance_cells):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("Choose a separate new or empty output directory for the calibrated recordings")
    metadata_path = source / "acquisition.json"
    acquisition = json.loads(metadata_path.read_text())
    layout = acquisition.get("sensor_layout")
    if layout not in ("antenna,direction,timestep", "direction,antenna,timestep"):
        raise ValueError("Expected explicit antenna, direction and timestep axes")
    measured_path = source / "sensor_samples.npy"
    reference_path = source / "empty_sensor_samples.npy"
    measured = np.load(measured_path, allow_pickle=False)
    reference = np.load(reference_path, allow_pickle=False)
    if layout == "direction,antenna,timestep":
        measured, reference = measured.swapaxes(0, 1), reference.swapaxes(0, 1)
    corrected, report = calibrate_pair(measured, reference, acquisition, clearance_cells)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "calibrated"
        staged.mkdir()
        copyfile(metadata_path, staged / "acquisition.json")
        copyfile(measured_path, staged / "sensor_samples.npy")
        if all(gain == 1. for gain in report["gain_by_receiver"]):
            copyfile(reference_path, staged / "empty_sensor_samples.npy")
        else:
            if layout == "direction,antenna,timestep":
                corrected = corrected.swapaxes(0, 1)
            np.save(staged / "empty_sensor_samples.npy", corrected, allow_pickle=False)
        (staged / "reference_calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Raw scene directory with the paired recordings and acquisition.json")
    parser.add_argument("destination", type=Path, help="New or empty calibrated scene directory")
    parser.add_argument("--clearance-cells", type=int, required=True, help=CLEARANCE_HELP)
    args = parser.parse_args()
    print(json.dumps(calibrate_scene(args.source, args.destination, args.clearance_cells), indent=2))


if __name__ == "__main__":
    main()
