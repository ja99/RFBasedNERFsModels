import math
import random
from dataclasses import dataclass

import torch
from torch import nn

def extract_params_from_network(model: nn.Module) -> torch.Tensor:
    params = []
    for param in model.parameters():
        params.append(param.flatten())
    return torch.cat(params)



def fill_network_with_params(model: nn.Module, params: torch.Tensor):
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param.data = params[start:end].reshape(param.shape)
        start = end


def create_3d_grid_indices(m:int, n:int, o:int)->torch.Tensor:
    """
    Create a PyTorch tensor containing all indices in an m x n x o grid.

    Args:
    m (int): Size of the first dimension.
    n (int): Size of the second dimension.
    o (int): Size of the third dimension.

    Returns:
    torch.Tensor: A tensor of shape (m*n*o, 3), where each row contains the indices (i, j, k).
    """
    i = torch.arange(m).view(-1, 1, 1).repeat(1, n, o)  # Repeat indices for i
    j = torch.arange(n).view(1, -1, 1).repeat(m, 1, o)  # Repeat indices for j
    k = torch.arange(o).view(1, 1, -1).repeat(m, n, 1)  # Repeat indices for k

    indices = torch.stack((i, j, k), dim=3)
    return indices.view(-1, 3)  # Reshape to have each triplet of indices as a row


def sample_indices(t: torch.Tensor, n_indices: int, n_bins: int = 10, by_index = None) -> torch.Tensor:
    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=t.device)
    indices = []

    if by_index is None:
        # Randomly choose a dimension to sample from
        dim_to_sample = random.choice([0, 1, 2])
    else:
        dim_to_sample = by_index

    # Assign each entry to a bin
    bins = torch.bucketize(t[:, dim_to_sample].contiguous(), bin_edges, right=True) - 1
    bins[bins == n_bins] = n_bins - 1  # Adjust for the rightmost edge

    n_per_bin = max(1, math.ceil(n_indices / n_bins))  # Ensure at least one sample per bin

    # Calculate bin frequencies
    bin_counts = torch.bincount(bins, minlength=n_bins)
    bin_frequencies = bin_counts.float() / bin_counts.sum()

    # Inverse frequencies for sampling from less common bins
    inverse_frequencies = 1.0 / (bin_frequencies + 1e-6)  # Avoid division by zero
    inverse_frequencies[bin_frequencies == 0] = 0  # Do not sample from empty bins
    sampling_weights = inverse_frequencies / inverse_frequencies.sum()

    for i in range(n_bins):
        bin_indices = torch.where(bins == i)[0]
        if 0 < len(bin_indices):
            selected = bin_indices[torch.randint(0, len(bin_indices), size=(n_per_bin,))]
            indices.append(selected)
        else:
            # Sample from less common bins based on inverse frequencies
            choice_bins = torch.multinomial(sampling_weights, n_per_bin, replacement=True)
            choice_indices = []
            for choice_bin in choice_bins:
                choice_bin_indices = torch.where(bins == choice_bin)[0]
                # Randomly choose an index from the selected bin
                if len(choice_bin_indices) > 0:
                    choice_index = choice_bin_indices[torch.randint(0, len(choice_bin_indices), (1,))]
                    choice_indices.append(choice_index)
            if choice_indices:
                indices.append(torch.cat(choice_indices))

    indices = torch.cat(indices) if indices else torch.tensor([], device=t.device, dtype=torch.long)

    # Shuffle the final indices to avoid ordering by bin
    if len(indices) > 0:
        indices = indices[torch.randperm(len(indices))]

    return indices[:n_indices]  # Ensure we return exactly n_indices
