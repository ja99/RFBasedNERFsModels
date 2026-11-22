"""Measure recovery of face-connected dielectric components against voxel masks."""

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import linear_sum_assignment


def component_counts(prediction, target, connectivity=6):
    """Count one-to-one matches with the same voxel adjacency on both masks."""
    if connectivity not in (6, 26):
        raise ValueError('Expected 6- or 26-connected voxel adjacency')
    neighbors = generate_binary_structure(3, 1 if connectivity == 6 else 3)
    predicted, n_pred = label(prediction, structure=neighbors)
    actual, n_true = label(target, structure=neighbors)
    table = np.bincount((actual * (n_pred + 1) + predicted).ravel(),
                        minlength=(n_true+1)*(n_pred+1)).reshape(n_true+1, n_pred+1)
    intersection = table[1:, 1:]
    union = table[1:].sum(1)[:, None] + table[:, 1:].sum(0)[None, :] - intersection
    iou = intersection / np.maximum(union, 1)
    result = {"target_components": n_true, "predicted_components": n_pred}
    for threshold in (.25, .50):
        eligible = iou >= threshold
        # Maximize eligible one-to-one matches, using IoU to break ties.
        left, right = linear_sum_assignment(-((min(n_true, n_pred) + 1) * eligible + iou))
        result[f"matched_components_iou{round(100*threshold)}"] = int(eligible[left, right].sum())
    return result


def summarize_components(rows):
    predicted = sum(r["predicted_components"] for r in rows)
    target = sum(r["target_components"] for r in rows)
    result = {"target_components": target, "predicted_components": predicted,
              "correct_component_count_scenes": sum(r["target_components"] == r["predicted_components"] for r in rows)}
    for threshold in (25, 50):
        key = f"matched_components_iou{threshold}"
        matched = sum(r[key] for r in rows)
        result[key] = matched
        result[f"component_recall_iou{threshold}"] = matched / target if target else None
        result[f"component_precision_iou{threshold}"] = matched / predicted if predicted else None
    return result
