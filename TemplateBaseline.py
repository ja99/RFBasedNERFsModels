"""Check whether similar calibrated RF recordings correspond to similar material geometry."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from CustomDataset import SensorDataset
from Visualizer import sdf_metrics
from field_training import check_preprocessing


def _signals(directory):
    return SensorDataset(directory, load_subsampled_grids=False)


def _labels(directory):
    return SensorDataset(directory, load_normal_grids=True, load_sensor_samples=False)


def evaluate_templates(training_dir, evaluation_dirs):
    train = _signals(training_dir)
    if train.preprocessing_id is None or train.waveform_mode != "calibrated":
        raise ValueError("Template matching requires a calibrated archive with a saved training split")
    indices = train.splits["train"]
    if not indices:
        raise ValueError("The template archive contains no training scenes")
    templates = F.normalize(torch.stack([train[i] for i in indices]), dim=1)
    train_grids = _labels(training_dir)
    rows = []
    for directory in evaluation_dirs:
        signals = _signals(directory)
        check_preprocessing(signals, train.waveform_mode, train.preprocessing_id)
        grids = _labels(directory)
        for i in signals.splits["test"]:
            waveform = signals[i]
            similarities = templates @ F.normalize(waveform, dim=0)
            best = int(similarities.argmax())
            template_index = indices[best]
            scores = sdf_metrics(train_grids[template_index][:, 2], grids[i][:, 2], grids.sdf_zero, grids.sdf_scale)
            rows.append(dict(scene=signals.scene_ids[i], template=train.scene_ids[template_index],
                             cosine=float(similarities[best]), **scores))
    if not rows:
        raise ValueError("No held-out scenes to compare")
    union = sum(row["occupied_union"] for row in rows)
    summary = dict(test_scenes=len(rows), mean_scene_iou=sum(r["occupancy_iou"] for r in rows) / len(rows),
                   mean_sdf_mae_cells=sum(r["sdf_mae_cells"] for r in rows) / len(rows),
                   pooled_iou=sum(r["occupied_intersection"] for r in rows) / union if union else 1.)
    return dict(scenes=rows, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path, help="archive directory providing the training templates")
    parser.add_argument("evaluation", type=Path, nargs="*", help="archive directories providing held-out scenes")
    args = parser.parse_args()
    print(json.dumps(evaluate_templates(args.training, args.evaluation or [args.training]), indent=2))
