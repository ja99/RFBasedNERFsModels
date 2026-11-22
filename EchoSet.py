"""Infer one or two separated laboratory inclusions from the calibrated RF measurements."""

import argparse
import json
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.ndimage import label
import torch
from torch import nn
from torch.nn import functional as F

from EchoKernel import DIRECTIONS, _archive, _calibrated_pair, _echo_features, _fit_projection, _mask, _project, _sdf
from EchoKernel import _settings, _signal, _target
from echo_components import component_counts, summarize_components


# Spheres and cuboids are centrally symmetric. Opposite support directions
# share a width, leaving the center free to move independently of shape.
NORMALS = DIRECTIONS[[n[np.flatnonzero(n)[0]] > 0 for n in DIRECTIONS]]
NORMAL_INDEX = np.abs(DIRECTIONS @ NORMALS.T).argmax(1)
C = 299792458.
INSTANCE_NEIGHBORS = np.ones((3, 3, 3), dtype=bool)


def components(mask):
    # A voxelized rotated box can have an edge- or corner-connected satellite.
    # The paired phantom generator separates different objects' bounds by >1 cell.
    labels, count = label(mask, structure=INSTANCE_NEIGHBORS)
    if count not in (1, 2):
        raise ValueError(f'Training requires one or two 26-connected inclusions, found {count}')
    result = np.zeros((2, 5 + len(NORMALS)), dtype='f4')
    for i in range(count):
        xyz = np.argwhere(labels == i + 1).astype('f4')
        center = xyz.mean(0)
        radius = (3 * len(xyz) / (4 * np.pi)) ** (1/3)
        width = np.ptp((xyz - center) @ NORMALS.T, axis=0) / 2
        result[i] = np.r_[1., center, np.log(radius), np.log((width + .35) / radius)]
    return result


def read_split(f, split, settings):
    ids = np.asarray(f['splits/' + split], dtype=int)
    if not len(ids):
        raise ValueError(f'The {split} split is empty')
    waves, masks, objects = [], [], []
    for i in ids:
        waves.append(_signal(f, i, settings))
        mask = _target(f, i) < 0
        masks.append(mask)
        objects.append(components(mask))
    return (np.stack(waves), np.stack(masks), np.stack(objects),
            [f['scene_ids'].asstr()[i] for i in ids])


def _grid(shape):
    seed = tuple(min(n, max(2, min(13, (n - 1)//3 + 1))) for n in shape)
    axes = [np.linspace(0, n-1, k, dtype='f4') for n, k in zip(shape, seed)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    return seed, points


def _travel_bank(settings, points):
    """Linear interpolation at the two-leg vacuum time for each transmitter, probe and beam."""
    meta = settings['acquisition']
    keys = ('antenna_positions_cells', 'transmit_positions_cells', 'grid_origin_cells',
            'frequency_hz', 'grid_spacing_m', 'time_step_s', 'sample_stride', 'beamforming_angles_deg')
    if not all(k in meta for k in keys):
        return None
    a, b, t = settings['sensor_shape']
    tx = np.asarray(meta['transmit_positions_cells'], dtype='f4')
    rx = (tx + np.asarray(meta['antenna_positions_cells'], dtype='f4')) / 2
    angles = np.deg2rad(meta['beamforming_angles_deg'])
    if tx.shape != (a, 3) or rx.shape != tx.shape or angles.shape != (b, 2):
        raise ValueError('Acquisition metadata and recorded aperture dimensions differ')
    az, el = angles.T
    direction = np.stack((np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)), axis=1)
    time_per_cell = meta['grid_spacing_m'] / (C * meta['time_step_s'] * meta['sample_stride'])
    delay = tx @ direction.T
    delay = (delay - delay.min(0)) * time_per_cell
    xyz = points + np.asarray(meta['grid_origin_cells'])
    rtx = np.linalg.norm(xyz[:, None] - tx, axis=-1)
    rrx = np.linalg.norm(xyz[:, None] - rx, axis=-1)
    bank = np.zeros((len(points), a, b, t), dtype='f4')
    g = np.arange(len(points))
    for j, beam, i in product(range(a), range(b), range(a)):
        arrival = (rtx[:, i] + rrx[:, j]) * time_per_cell + delay[i, beam]
        start = np.floor(arrival).astype(int)
        ok = (start >= 0) & (start < t-1)
        w = 1. / ((rtx[ok, i] + 2.) * (rrx[ok, j] + 2.))
        fraction = arrival[ok] - start[ok]
        bank[g[ok], j, beam, start[ok]] += w * (1-fraction)
        bank[g[ok], j, beam, start[ok]+1] += w * fraction
    bank = bank.reshape(len(points), -1)
    bank /= np.maximum(bank.sum(1, keepdims=True), 1e-12)
    return np.ascontiguousarray(bank.T)


class Features:
    def __init__(self, settings, transforms, device):
        self.settings, self.transforms, self.device = settings, transforms, device
        self.seed, self.points = _grid(settings['grid_shape'])
        bank = _travel_bank(settings, self.points)
        self.bank = None if bank is None else torch.from_numpy(bank).to(device)
        if self.bank is not None:
            acq = settings['acquisition']
            period = 1. / (acq['frequency_hz'] * acq['time_step_s'] * acq['sample_stride'])
            k = min(settings['sensor_shape'][-1], max(3, int(np.ceil(period))))
            phase = torch.arange(k, device=device) * (2*np.pi/period)
            self.pulse = torch.stack((phase.sin(), phase.cos(), torch.ones_like(phase)))[:, None] / k

    def _image(self, waves):
        if self.bank is None:
            return torch.zeros(len(waves), 3, *self.seed, device=self.device)
        outputs = []
        for batch in torch.from_numpy(waves).split(48):
            x = batch.to(self.device)
            rms = x.square().mean(-1, keepdim=True).sqrt()
            floor = rms.amax((1, 2, 3), keepdim=True).clamp_min(1e-12) * 1e-3
            relative = x / rms.clamp_min(floor).pow(.65)
            relative /= relative.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
            k = self.pulse.size(-1)
            traces = relative.reshape(-1, 1, x.size(-1))
            padded = F.pad(traces, (0, k-1))
            matched = F.conv1d(padded, self.pulse[:2])
            energy = F.conv1d(padded.square(), self.pulse[2:]).sqrt()
            filtered = torch.cat((matched, energy), dim=1).reshape(*x.shape[:3], 3, x.size(-1))
            image = filtered.permute(0, 3, 1, 2, 4).flatten(2) @ self.bank
            outputs.append(image.reshape(len(x), 3, *self.seed))
        return torch.cat(outputs)

    @staticmethod
    def groups(waves):
        return [np.stack(items) for items in zip(*(_echo_features(wave) for wave in waves))]

    def encode(self, waves, groups=None):
        groups = self.groups(waves) if groups is None else groups
        compressed = np.concatenate(_project(groups, self.transforms), axis=1)
        return torch.from_numpy(compressed).to(self.device), self._image(waves)


class EchoSet(nn.Module):
    def __init__(self, code_size, shape, hidden=192, shape_rank=12, landmarks=128):
        super().__init__()
        self.config = dict(code_size=code_size, shape=list(shape), hidden=hidden,
                           shape_rank=shape_rank, landmarks=landmarks)
        self.seed, points = _grid(shape)
        g = len(points)
        self.register_buffer('points', torch.from_numpy(points))
        self.register_buffer('extents', torch.tensor(shape, dtype=torch.float32).sub(1).clamp_min(1))
        self.register_buffer('normals', torch.from_numpy(NORMALS.copy()))
        self.register_buffer('shape_mean', torch.zeros(len(NORMALS)))
        self.register_buffer('shape_basis', torch.zeros(shape_rank, len(NORMALS)))
        self.register_buffer('radius_stats', torch.tensor([1.2, .3]))
        self.register_buffer('proposal_spacing', torch.tensor(4.5))
        self.register_buffer('anchors', torch.zeros(landmarks, code_size))
        self.register_buffer('prior_width', torch.ones(()))
        self.register_buffer('prior_map', torch.zeros(code_size+landmarks+1, g))
        self.register_buffer('image_mean', torch.zeros(3, *self.seed))
        self.register_buffer('image_scale', torch.ones(3, *self.seed))
        self.global_encoder = nn.Sequential(nn.Linear(code_size, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.seed_features = nn.Linear(hidden, 6*g)
        self.spatial = nn.Sequential(nn.Conv3d(10, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv3d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU())
        self.heat = nn.Conv3d(24, 1, 1)
        self.count = nn.Sequential(nn.Linear(hidden + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.slot = nn.Sequential(nn.Linear(hidden+28, 128), nn.LayerNorm(128), nn.SiLU(),
                                  nn.Linear(128, 5+shape_rank))
        nn.init.zeros_(self.heat.weight)
        nn.init.constant_(self.heat.bias, -3.8)
        nn.init.normal_(self.slot[-1].weight, std=.01)
        nn.init.zeros_(self.slot[-1].bias)

    def design(self, x):
        radial = torch.exp(-torch.cdist(x, self.anchors).square() / self.prior_width)
        return torch.cat((x, radial, torch.ones_like(x[:, :1])), 1)

    def center_map(self, objects):
        valid = objects[:, :, 0].bool()
        distance = torch.cdist(objects[:, :, 1:4], self.points)
        width = (objects[:, :, 4].exp() * .45).clamp_min(1.6)
        heat = torch.exp(-.5*(distance / width[:, :, None]).square())
        nearest = distance.argmin(-1)
        heat.scatter_(-1, nearest[:, :, None], 1.)
        return (heat * valid[:, :, None]).amax(1), nearest

    @torch.no_grad()
    def initialize(self, x, images, objects, seed):
        valid = objects[..., 0].bool()
        radii = objects[..., 4][valid]
        self.radius_stats.copy_(torch.stack((radii.mean(), radii.std(unbiased=False).clamp_min(.15))))
        pairs = valid.all(1)
        if int(pairs.sum()) >= 8:
            separation = (objects[pairs, 0, 1:4] - objects[pairs, 1, 1:4]).norm(dim=1)
            self.proposal_spacing.copy_((.65*torch.quantile(separation, .05)).clamp_min(2.))
        else:
            self.proposal_spacing.copy_((1.25*radii.exp().median()).clamp_min(2.))
        shapes = objects[..., 5:][valid]
        self.shape_mean.copy_(shapes.mean(0))
        values, vectors = torch.linalg.eigh(torch.cov(shapes.T) if len(shapes)>1 else torch.eye(len(NORMALS),device=x.device)*.001)
        rank = self.shape_basis.size(0)
        self.shape_basis.copy_((values[-rank:].clamp_min(.001).sqrt()[None] * vectors[:, -rank:]).T)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        ids = torch.randperm(len(x), generator=gen)[:len(self.anchors)].to(x.device)
        self.anchors.copy_(x[ids])
        self.prior_width.copy_(torch.pdist(self.anchors).square().median().clamp_min(1e-4)
                               if len(ids)>1 else x.new_tensor(1.))
        self.image_mean.copy_(images.mean(0))
        spread = images.std(0, unbiased=False)
        self.image_scale.copy_(spread.clamp_min(spread.flatten(1).median(1).values[:, None, None, None]*.2 + 1e-5))
        target = self.center_map(objects)[0]
        design = self.design(x).double()
        gram = design.T @ design
        gram.diagonal().add_(max(.02, float(gram.diagonal().mean()) * .012))
        self.prior_map.copy_(torch.linalg.solve(gram, design.T @ target.double()).float())

    def encode(self, x, images):
        hidden = self.global_encoder(x)
        prior = (self.design(x) @ self.prior_map).reshape(-1, 1, *self.seed)
        images = ((images - self.image_mean)/self.image_scale).clamp(-6, 6)
        raw = self.seed_features(hidden).reshape(-1, 6, *self.seed)
        spatial = self.spatial(torch.cat((images, prior, raw), dim=1))
        heat = (self.heat(spatial) + 7*prior).flatten(1)
        first = heat.argmax(1)
        separation = (self.points[None]-self.points[first, None]).square().sum(-1)
        # Resolve two local peaks on the output grid before predicting fine shapes.
        maxima = F.max_pool3d(heat.reshape(-1, 1, *self.seed), 3, stride=1, padding=1).flatten(1)
        far = separation >= self.proposal_spacing.square()
        separate_peak = (heat >= maxima) & far
        allowed = torch.where(separate_peak.any(1, keepdim=True), separate_peak, far)
        alternate = heat.masked_fill(~allowed, -1e4).argmax(1)
        indices = torch.stack((first, alternate), 1)
        strengths = heat.gather(1, indices)
        distance = separation.gather(1, alternate[:, None]).sqrt() / self.extents.norm()
        count = self.count(torch.cat((hidden, strengths, distance, heat.mean(1, keepdim=True)), 1)).squeeze(1)
        return hidden, spatial.flatten(2).transpose(1, 2), heat, indices, count

    def decode(self, state, indices=None):
        hidden, spatial, heat, peaks, count = state
        indices = peaks if indices is None else indices
        batch = torch.arange(len(hidden), device=hidden.device)[:, None]
        anchor = self.points[indices]
        features = torch.cat((hidden[:, None].expand(-1, 2, -1), spatial[batch, indices],
                              anchor/self.extents-.5, heat[batch, indices, None]), -1)
        raw = self.slot(features)
        center = anchor + 5.*raw[:, :, :3].tanh()
        logr = (self.radius_stats[0] + self.radius_stats[1]*raw[:, :, 3]).clamp(-1., self.extents.max().log())
        ratio = (self.shape_mean + raw[:, :, 5:] @ self.shape_basis).clamp(-2., 2.)
        widths = ((ratio + logr[:, :, None]).exp() - .35).clamp_min(.08)
        return center, logr, ratio, widths, raw[:, :, 4]

    def teacher_proposals(self, state, objects):
        # Jitter annotated centers while learning sub-cell refinement; for a
        # single object the other proposal is the strongest distant RF peak.
        target = objects[:, :, 1:4] + torch.randn_like(objects[:, :, 1:4])*1.35
        ids = torch.cdist(target, self.points).argmin(-1)
        rows = torch.arange(len(ids), device=ids.device)
        first_valid = objects[:, :, 0].argmax(1)
        single_center = objects[rows, first_valid, 1:4]
        far = torch.cdist(single_center[:, None], self.points).squeeze(1) > self.proposal_spacing
        negative = state[2].masked_fill(~far, -1e4).argmax(-1)
        dual = objects[:, :, 0].sum(1) > 1
        ids[:, 0] = torch.where(dual, ids[:, 0], ids[rows, first_valid])
        ids[:, 1] = torch.where(dual, ids[:, 1], negative)
        return ids


def assignment_loss(decoded, objects):
    center, logr, ratio, unused, existence = decoded
    active = objects[:, None, :, 0]
    center_delta = (center[:, :, None] - objects[:, None, :, 1:4]).norm(dim=-1)
    center_cost = (torch.sqrt(1 + center_delta.square()) - 1) / objects[:, None, :, 4].exp().clamp_min(1.5)
    radius_cost = (logr[:, :, None] - objects[:, None, :, 4]).abs()
    shape_cost = (ratio[:, :, None] - objects[:, None, :, 5:]).abs().mean(-1)
    present_cost = F.softplus(existence[:, :, None]) - active * existence[:, :, None]
    pair = active*(2.*center_cost + radius_cost + 1.3*shape_cost) + .45*present_cost
    direct = pair[:, 0, 0] + pair[:, 1, 1]
    swapped = pair[:, 0, 1] + pair[:, 1, 0]
    return (torch.minimum(direct, swapped) / active.sum(-1).flatten().clamp_min(1)).mean()


def _query_loss(model, decoded, objects, masks, queries):
    center, unused_logr, unused_ratio, widths, existence = decoded
    b = len(objects)
    device = objects.device
    # Sample small-object neighborhoods as well as the full scene.
    active = objects[:, :, 0]
    ids = torch.multinomial(active / active.sum(1, keepdim=True), queries, replacement=True)
    rows = torch.arange(b, device=device)[:, None]
    around = objects[rows, ids]
    query = around[:, :, 1:4] + ((2*torch.rand(b, queries, 3, device=device)-1)
                                  * 1.65 * around[:, :, 4:5].exp())
    global_count = max(1, queries//4)
    query[:, :global_count] = torch.rand(b, global_count, 3, device=device) * model.extents
    query = torch.minimum(query.round().clamp_min(0), masks.new_tensor(masks.shape[1:], dtype=torch.float32)-1)
    ix, iy, iz = query.long().unbind(-1)
    truth = masks[rows, ix, iy, iz]
    delta = query[:, None] - center[:, :, None]
    margin = ((delta @ model.normals.T).abs() - widths[:, :, None]).amax(-1)
    probability = (-margin/.55).sigmoid() * existence.sigmoid()[:, :, None]
    union = 1 - (1-probability).prod(dim=1)
    error = F.binary_cross_entropy(union.clamp(1e-6, 1-1e-6), truth.float(), reduction='none')
    positives = (error*truth).sum(1) / truth.sum(1).clamp_min(1)
    negatives = (error*~truth).sum(1) / (~truth).sum(1).clamp_min(1)
    return (positives + negatives).mean()/2


def _train_loss(model, x, img, objects, masks, teacher_rate, queries):
    state = model.encode(x, img)
    heat, _ = model.center_map(objects)
    probability = state[2].sigmoid()
    positive = heat >= 1.-1e-6
    ce = (F.softplus(-state[2]) * (1-probability).square() * positive
          + F.softplus(state[2]) * probability.square() * (1-heat).pow(4) * ~positive)
    heat_loss = ce.sum() / positive.sum().clamp_min(1)
    use_teacher = torch.rand(len(x), 1, device=x.device) < teacher_rate
    ids = torch.where(use_teacher, model.teacher_proposals(state, objects), state[3])
    decoded = model.decode(state, ids)
    dual = objects[:, :, 0].sum(1) > 1
    count_loss = F.binary_cross_entropy_with_logits(state[4], dual.float())
    shape_loss = assignment_loss(decoded, objects)
    # The generated double phantoms have disjoint axis-aligned bounding boxes.
    # Apply this penalty only when that condition holds in the training label.
    axis = torch.as_tensor(NORMAL_INDEX[[0, 2, 4]], device=x.device)
    widths = decoded[3][:, :, axis]
    gap = ((decoded[0][:, 0] - decoded[0][:, 1]).abs() - widths.sum(1)).amax(1)
    actual_width = (objects[:, :, 5:][:, :, axis] + objects[:, :, 4:5]).exp() - .35
    actual_gap = ((objects[:, 0, 1:4] - objects[:, 1, 1:4]).abs() - actual_width.sum(1)).amax(1)
    separated = dual & (actual_gap >= 1.)
    gap_loss = (F.relu(1. - gap)*separated).sum() / separated.sum().clamp_min(1)
    return heat_loss + shape_loss + .65*count_loss + .22*_query_loss(model, decoded, objects, masks, queries) + .08*gap_loss


@torch.inference_mode()
def _predict(model, encoded, batch):
    x, images = encoded
    outputs = []
    model.eval()
    for low in range(0, len(x), batch):
        state = model.encode(x[low:low+batch], images[low:low+batch])
        center, unused_r, unused_shape, widths, existence = model.decode(state)
        strength = existence + .25*state[2].gather(1, state[3])
        count = state[4].sigmoid()
        arrays = [z.cpu().numpy() for z in (center, widths, strength, count)]
        outputs.extend(zip(*arrays))
    return outputs


def _slot_masks(prediction, shape, offset):
    center, widths, strength, two = prediction
    order = np.argsort(-strength)
    supports = center @ DIRECTIONS.T + widths[:, NORMAL_INDEX]
    masks = [_mask(supports[i], shape, offset) for i in order]
    return masks, supports[order], center[order], two


def _iou(a, b):
    both = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return both, union, both/union if union else 1.


def _select(predictions, masks, objects):
    actual_count = objects[:, :, 0].sum(1)
    probs = np.array([p[-1] for p in predictions])
    best = None
    for offset in (0., -.4, .4):
        ious, counts = [], []
        for pred, truth in zip(predictions, masks):
            slots, unused_h, unused_c, unused_two = _slot_masks(pred, truth.shape, offset)
            single, double = slots[0], slots[0] | slots[1]
            ious.append([_iou(single, truth)[2], _iou(double, truth)[2]])
            counts.append([label(single, INSTANCE_NEIGHBORS)[1], label(double, INSTANCE_NEIGHBORS)[1]])
        ious, counts = np.asarray(ious), np.asarray(counts)
        for threshold in (.50, .35, .65, .22, .78):
            second = (probs >= threshold).astype(int)
            rows = np.arange(len(masks))
            iou = ious[rows, second]
            correct = counts[rows, second] == actual_count
            accuracy = float(np.mean([correct[actual_count==n].mean() for n in np.unique(actual_count)]))
            score = .75*float(iou.mean()) + .25*accuracy
            if best is None or score > best['selection_score']:
                best = dict(selection_score=score, threshold=threshold, offset=offset,
                            mean_scene_iou=float(iou.mean()), balanced_count_accuracy=accuracy,
                            correct_component_count_scenes=int(correct.sum()), instance_connectivity=26, scenes=len(masks))
    return best


def _save(path, model, features, settings, prior, prior_id, training_ids, validation_ids, validation, epoch, seed):
    transforms = [[torch.from_numpy(value) for value in block] for block in features.transforms]
    value = dict(format_version=1, model_type='echo_set', config=model.config, weights=model.state_dict(),
                 transforms=transforms, settings=settings, prior=torch.from_numpy(prior), prior_id=prior_id,
                 training_ids=training_ids, validation_ids=validation_ids, validation=validation, epoch=epoch, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix='.echo-set-', suffix='.pt', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fit(data, output, *, device='cpu', epochs=125, batch_size=48, val_every=5, patience=35,
        seed=42, hidden=192, queries=112, learning_rate=.0013):
    if min(epochs, batch_size, val_every, patience, queries, hidden) < 1:
        raise ValueError('Training counts must be positive')
    torch.manual_seed(seed)
    device = torch.device(device)
    with _archive(data) as f:
        settings = _settings(f)
        tr, masks, obj, train_ids = read_split(f, 'train', settings)
        va, val_masks, val_obj, val_ids = read_split(f, 'validation', settings)
    if set(train_ids) & set(val_ids):
        raise ValueError('Training and validation scenes overlap')
    groups = Features.groups(tr)
    rng = np.random.default_rng(seed)
    transforms = [_fit_projection(g, i==2, rng) for i, g in enumerate(groups)]
    features = Features(settings, transforms, device)
    x, images = features.encode(tr, groups)
    val = features.encode(va)
    del groups, tr, va
    prior_frequency = masks.mean(0, dtype='f4')
    numerator = np.einsum('nijk,ijk->n', masks, prior_frequency, optimize=True)
    prior_idx = np.argmax(numerator/(masks.sum((1,2,3)) + prior_frequency.sum()-numerator).clip(min=1))
    prior, prior_id = masks[prior_idx].copy(), train_ids[prior_idx]
    masks = torch.from_numpy(masks).to(device)
    obj = torch.from_numpy(obj).to(device)
    model = EchoSet(x.size(1), settings['grid_shape'], hidden=hidden, landmarks=min(128, len(x))).to(device)
    model.initialize(x, images, obj, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=learning_rate*.10)
    best = -1.
    best_epoch = 0
    best_validation = None
    for epoch in range(1, epochs+1):
        model.train()
        teacher_rate = .18 + .60 * max(0., 1.-(epoch-1)/max(10., .45*epochs))
        for ids in torch.randperm(len(x), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = _train_loss(model, x[ids], images[ids], obj[ids], masks[ids], teacher_rate, queries)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite unordered set loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
        schedule.step()
        if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
            validation = _select(_predict(model, val, batch_size), val_masks, val_obj)
            print(json.dumps(dict(epoch=epoch, validation=validation)), flush=True)
            if validation['selection_score'] > best:
                best, best_epoch, best_validation = validation['selection_score'], epoch, validation
                _save(output, model, features, settings, prior, prior_id, train_ids, val_ids, validation, epoch, seed)
            if epoch-best_epoch >= patience:
                break
    return dict(training_scenes=len(train_ids), validation_scenes=len(val_ids), best_epoch=best_epoch,
                validation=best_validation, checkpoint=str(output), model_bytes=Path(output).stat().st_size)


class SetPredictor:
    def __init__(self, path, device='cpu'):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data.get('model_type') != 'echo_set' or data.get('format_version') != 1:
            raise ValueError('Expected an EchoSet checkpoint')
        self.settings = data['settings']
        self.selection = data['validation']
        self.prior, self.prior_id = data['prior'].numpy(), data['prior_id']
        self.training_ids = data['training_ids']
        transforms = [[tensor.numpy() for tensor in block] for block in data['transforms']]
        self.features = Features(self.settings, transforms, torch.device(device))
        self.model = EchoSet(**data['config'])
        self.model.load_state_dict(data['weights'])
        self.model.to(device).eval()

    def check(self, f):
        if _settings(f) != self.settings:
            raise ValueError('Archive acquisition and preprocessing differ from the trained set predictor')

    def fields(self, waves, batch_size=32):
        predictions = _predict(self.model, self.features.encode(waves), batch_size)
        results = []
        for prediction in predictions:
            slots, h, centers, p_two = _slot_masks(prediction, self.settings['grid_shape'], self.selection['offset'])
            n = 1 + int(p_two >= self.selection['threshold'])
            slots = np.stack(slots[:n])
            union = slots.any(0)
            sdf = _sdf(union, h[:n].max(0), self.selection['offset'])
            results.append((sdf, slots, centers[:n], float(p_two)))
        return results

    def predict_scene(self, directory):
        return self.fields(_calibrated_pair(directory, self.settings)[None])[0]


def _score(actual, predictions, names):
    rows = []
    for truth, predicted, name in zip(actual, predictions, names):
        target, mask = truth < 0, predicted < 0
        both, union, iou = _iou(mask, target)
        rows.append(dict(scene=name, occupancy_iou=iou, occupied_intersection=both, occupied_union=union,
                         sdf_mae_cells=float(np.abs(predicted-truth).mean()), **component_counts(mask, target),
                         physical_instances=component_counts(mask, target, connectivity=26)))

    def summary(part):
        total = sum(r['occupied_union'] for r in part)
        physical = [r['physical_instances'] for r in part]
        instance_accuracy = np.mean([p['predicted_components']==p['target_components'] for p in physical])
        return dict(scenes=len(part), mean_scene_iou=float(np.mean([r['occupancy_iou'] for r in part])),
                    pooled_iou=sum(r['occupied_intersection'] for r in part)/total if total else 1.,
                    mean_sdf_mae_cells=float(np.mean([r['sdf_mae_cells'] for r in part])),
                    component_count_accuracy=float(np.mean([r['predicted_components']==r['target_components'] for r in part])),
                    component_connectivity=6, **summarize_components(part),
                    physical_instances=dict(connectivity=26, count_accuracy=float(instance_accuracy),
                                            **summarize_components(physical)))
    report = summary(rows)
    report['by_target_components'] = {
        str(n): summary([r for r in rows if r['target_components']==n])
        for n in sorted({r['target_components'] for r in rows})}
    report['by_target_instances'] = {
        str(n): summary([r for r in rows if r['physical_instances']['target_components']==n])
        for n in sorted({r['physical_instances']['target_components'] for r in rows})}
    return dict(scenes=rows, summary=report)


def _nearest(predictor, data, waves):
    with _archive(data) as f:
        predictor.check(f)
        indices = np.asarray(f['splits/train'], dtype=int)
        if not len(indices):
            return None
        names = [f['scene_ids'].asstr()[i] for i in indices]
        if set(names) != set(predictor.training_ids):
            raise ValueError('Template control must use exactly the fitted set of training scenes')
        train = np.stack([_signal(f, i, predictor.settings).ravel() for i in indices])
        train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        best = (waves.reshape(len(waves), -1) @ train.T).argmax(1)
        return [_target(f, indices[i]) for i in best], [names[i] for i in best]


def evaluate(predictor, data, split='test', batch_size=32, controls=True, templates=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    with _archive(data) as f:
        predictor.check(f)
        indices = np.asarray(f['splits/'+split], dtype=int)
        if not len(indices):
            raise ValueError(f'No {split} scenes in archive')
        waves = np.stack([_signal(f, i, predictor.settings) for i in indices])
        truth = [_target(f, i) for i in indices]
        names = [f['scene_ids'].asstr()[i] for i in indices]
    estimated = predictor.fields(waves, batch_size)
    fields = [item[0] for item in estimated]
    result = _score(truth, fields, names)
    for row, item in zip(result['scenes'], estimated):
        row['selected_slots'] = len(item[1])
        row['probability_two'] = item[3]
    result['summary'].update(split=split, component_connectivity=6,
                             count_threshold=predictor.selection['threshold'], support_offset_cells=predictor.selection['offset'])
    if controls:
        fixed = _sdf(predictor.prior, np.zeros(len(DIRECTIONS), dtype='f4'), 0.)
        prior = _score(truth, [fixed]*len(truth), names)
        prior['template_scene'] = predictor.prior_id
        result['controls'] = dict(fixed_training_template=prior)
        if len(names) > 1:
            order = np.random.default_rng(1741).permutation(len(names))
            source = np.empty_like(order)
            source[order] = np.roll(order, 1)
            shuffled = _score(truth, [fields[i] for i in source], names)
            for row, i in zip(shuffled['scenes'], source):
                row['echo_scene'] = names[i]
            result['controls']['shuffled_rf'] = shuffled
        nearest = _nearest(predictor, templates or data, waves)
        if nearest is not None:
            retrieved = _score(truth, nearest[0], names)
            for row, scene in zip(retrieved['scenes'], nearest[1]):
                row['template_scene'] = scene
            result['controls']['cosine_nearest_training_template'] = retrieved
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    train = commands.add_parser('fit')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=125)
    train.add_argument('--batch-size', type=int, default=48)
    train.add_argument('--patience', type=int, default=35)
    train.add_argument('--val-every', type=int, default=5)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--hidden', type=int, default=192)
    train.add_argument('--queries', type=int, default=112)
    train.add_argument('--learning-rate', type=float, default=.0013)
    ev = commands.add_parser('evaluate')
    ev.add_argument('--data', type=Path, required=True)
    ev.add_argument('--model', type=Path, required=True)
    ev.add_argument('--split', choices=('train','validation','test'), default='test')
    ev.add_argument('--batch-size', type=int, default=32)
    ev.add_argument('--no-controls', action='store_true')
    ev.add_argument('--templates', type=Path, help='Supply the training archive when evaluating an external archive')
    ev.add_argument('--output', type=Path)
    infer = commands.add_parser('infer')
    infer.add_argument('--model', type=Path, required=True)
    inputs = infer.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--scene-dir', type=Path)
    inputs.add_argument('--data', type=Path)
    infer.add_argument('--scene-id')
    infer.add_argument('--output', type=Path, required=True)
    infer.add_argument('--instances', type=Path, help='Optional NPZ of the selected masks, predicted centers and count probability')
    for command in (train, ev, infer):
        command.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.action == 'fit':
        report = fit(**{key:value for key, value in vars(args).items() if key!='action'})
    elif args.action == 'evaluate':
        report = evaluate(SetPredictor(args.model, args.device), args.data, args.split, args.batch_size,
                          not args.no_controls, args.templates)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    else:
        predictor = SetPredictor(args.model, args.device)
        if args.scene_dir:
            if args.scene_id:
                parser.error('--scene-id goes with --data')
            field, slots, centers, p_two = predictor.predict_scene(args.scene_dir)
        else:
            with _archive(args.data) as f:
                predictor.check(f)
                names = f['scene_ids'].asstr()[:].tolist()
                if args.scene_id not in names:
                    parser.error('--data requires an existing --scene-id')
                wave = _signal(f, names.index(args.scene_id), predictor.settings)
            field, slots, centers, p_two = predictor.fields(wave[None])[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as handle:
            np.save(handle, field)
        if args.instances:
            args.instances.parent.mkdir(parents=True, exist_ok=True)
            with args.instances.open('wb') as handle:
                np.savez_compressed(handle, masks=slots, centers_cells=centers, probability_two=p_two)
        report = dict(output=str(args.output), units='grid cells', selected_slots=len(slots),
                      predicted_components=int(label(field<0)[1]), predicted_instances=int(label(field<0, INSTANCE_NEIGHBORS)[1]),
                      component_connectivity=6, instance_connectivity=26, probability_two=p_two)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
