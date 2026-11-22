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


def sample_indices(t: torch.Tensor, n_indices: int, n_bins: int = 10, by_index=None) -> torch.Tensor:
    """Balance the query budget across occupied target bins, without reusing available voxels."""
    if t.ndim != 2 or min(t.shape) == 0 or n_bins < 1 or n_indices < 1:
        raise ValueError("Expected a nonempty [voxel, channel] tensor and positive sample/bin counts")

    n_voxels, n_channels = t.shape
    rounds, remaining = divmod(n_indices, n_voxels)
    samples = [torch.randperm(n_voxels, device=t.device) for _ in range(rounds)]
    if remaining:
        channel = torch.randint(n_channels, ()).item() if by_index is None else by_index
        edges = torch.linspace(0, 1, n_bins + 1, dtype=t.dtype, device=t.device)
        bins = torch.bucketize(t[:, channel].contiguous(), edges[1:-1], right=True)
        capacities = torch.bincount(bins, minlength=n_bins).tolist()
        # Randomize who gets the final, indivisible part of the budget.
        active = [b for b in torch.randperm(n_bins).tolist() if capacities[b]]
        quotas = [0] * n_bins
        while remaining:
            share, extra = divmod(remaining, len(active))
            for i, b in enumerate(active):
                take = min(share + (i < extra), capacities[b] - quotas[b])
                quotas[b] += take
                remaining -= take
            active = [b for b in active if quotas[b] < capacities[b]]

        for b, count in enumerate(quotas):
            if count:
                members = (bins == b).nonzero(as_tuple=True)[0]
                order = torch.randperm(len(members), device=t.device)[:count]
                samples.append(members[order])

    indices = torch.cat(samples)
    return indices[torch.randperm(n_indices, device=t.device)]


def uniform_query_indices(n_scenes, n_voxels, n_queries, device, *, seed=None):
    """Draw once from each of evenly spaced voxel ranges per scene.

    Splitting the flattened volume into disjoint ranges covers its depth and keeps
    memory proportional to the query budget. A local generator leaves training RNG
    state untouched when validation repeats on the same ordered batches.
    """
    if min(n_scenes, n_voxels, n_queries) < 1:
        raise ValueError("Scene, voxel and query counts must be positive")
    count = min(n_queries, n_voxels)
    bounds = torch.arange(count + 1, dtype=torch.long, device=device) * n_voxels // count
    if seed is None:
        offsets = torch.rand(n_scenes, count, device=device)
    else:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        offsets = torch.rand(n_scenes, count, device="cpu", generator=rng).to(device)
    return bounds[:-1] + (offsets * bounds.diff()).long().minimum(bounds.diff() - 1)
