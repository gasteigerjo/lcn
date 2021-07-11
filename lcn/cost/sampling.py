from functools import partial

import torch

from lcn.utils import get_matching_embeddings


def get_other_sample(X, C, p, dist, min_dist=1e-5):
    D_ij = dist.cdist(X, C)  # (batch_size, 2*max_points, kclusters)
    mask = torch.any(D_ij < min_dist, dim=-1)
    p.masked_fill_(mask, 0)
    rows = torch.multinomial(p, 1).squeeze()
    batch_idx = torch.arange(X.shape[0], dtype=torch.long, device=X.device)
    return X[batch_idx, rows]


@torch.no_grad()
def proportional_sampling(X, p, ksamples, dist):
    batch_size, _, _ = X.shape

    landmark_idx = torch.multinomial(p, ksamples, replacement=False)
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=X.device)[
        :, None
    ].expand_as(landmark_idx)
    samples = X[batch_idx, landmark_idx]

    # Resample to prevent matching samples
    batch_matching, idx_matching1, _ = get_matching_embeddings(samples, dist=dist)
    while len(batch_matching) > 0:
        samples[batch_matching, idx_matching1] = get_other_sample(
            X[batch_matching], samples[batch_matching], p[batch_matching], dist=dist
        )
        batch_matching, idx_matching1, _ = get_matching_embeddings(samples, dist=dist)
    return samples


@torch.no_grad()
def uniform_sampling_cat(X, num_points, ksamples, dist):
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2
    assert torch.all(num_points.sum(0) >= ksamples)

    mask_points = torch.zeros(
        (batch_size, 2 * max_points), dtype=torch.bool, device=X.device
    )
    mask_points[:, :max_points] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[0, :, None]
    )
    mask_points[:, max_points:] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[1, :, None]
    )
    probs = mask_points.float()

    return proportional_sampling(X, probs, ksamples, dist)


@torch.no_grad()
def norm_sampling_cat(
    X, num_points, ksamples, dist, norm_fn=partial(torch.norm, p=2, dim=-1)
):
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2
    assert torch.all(num_points.sum(0) >= ksamples)

    probs = norm_fn(X)

    mask_points = torch.zeros(
        (batch_size, 2 * max_points), dtype=torch.bool, device=X.device
    )
    mask_points[:, :max_points] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[0, :, None]
    )
    mask_points[:, max_points:] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[1, :, None]
    )
    probs *= mask_points

    return proportional_sampling(X, probs, ksamples, dist)


@torch.no_grad()
def norm_choice_cat(X, num_points, ksamples, norm_fn=partial(torch.norm, p=2, dim=-1)):
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2
    assert torch.all(num_points.sum(0) >= ksamples)

    weights = norm_fn(X)

    mask_points = torch.zeros(
        (batch_size, 2 * max_points), dtype=torch.bool, device=X.device
    )
    mask_points[:, :max_points] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[0, :, None]
    )
    mask_points[:, max_points:] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        < num_points[1, :, None]
    )
    weights *= mask_points

    landmark_idx = torch.topk(weights, k=ksamples, dim=-1, sorted=False).indices
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=X.device)[
        :, None
    ].expand_as(landmark_idx)
    samples = X[batch_idx, landmark_idx]

    return samples


def get_kmeanspp_sample(X, C, mask, dist):
    D_ij = dist.cdist(X, C)  # (batch_size, 2*max_points, kclusters)
    min_dists = D_ij.min(
        dim=2
    ).values  # (batch_size, 2*max_points) Dist to nearest cluster
    min_dists.masked_fill_(mask, 0)
    probs = min_dists ** 2
    probs[(probs.sum(-1) == 0), :] = 1
    rows = torch.multinomial(probs, 1).squeeze()
    batch_idx = torch.arange(X.shape[0], dtype=torch.long, device=X.device)
    return X[batch_idx, rows]


def add_kmeanspp_sample(X, C, dist_XC, isample, mask, dist):
    """
    Sample initial centroids according to k-means++.

    If there is an "Assertion `val >= zero` failed." error in this method, this is due to the multinomial.
    This is because of NaNs in X, which are caused by failing backprop due to some instabilities.
    Debug this by setting `torch.autograd.set_detect_anomaly(True)`.
    Unfortunately, approximate Sinkhorn is sometimes unstable
    if the regularization is too low or there are very few landmarks.
    For example, GTN with LCN-Sinkhorn and without the BP matrix can fail here.
    """
    dist_XC[:, :, isample - 1] = dist.cdist(X, C[:, isample - 1][:, None]).squeeze(
        dim=-1
    )  # (batch_size, 2*max_points, 1)
    min_dists = (
        dist_XC[:, :, :isample].min(dim=2).values
    )  # (batch_size, 2*max_points) Dist to nearest cluster
    min_dists.masked_fill_(mask, 0)
    probs = min_dists ** 2
    probs[(probs.sum(-1) == 0), :] = 1
    rows = torch.multinomial(probs, 1, replacement=True).squeeze()
    batch_idx = torch.arange(X.shape[0], dtype=torch.long, device=X.device)
    C[:, isample] = X[batch_idx, rows]


@torch.no_grad()
def kmeanspp_sampling_cat(X, ksamples, dist, num_points=None, mask_padding=None):
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2
    if num_points is not None:
        assert torch.all(num_points.sum(0) >= ksamples)

    if mask_padding is None:
        assert num_points is not None
        mask_padding = torch.zeros(
            (batch_size, 2 * max_points), dtype=torch.bool, device=X.device
        )
        mask_padding[:, :max_points] = (
            torch.arange(max_points, dtype=torch.long, device=X.device).expand(
                batch_size, -1
            )
            >= num_points[0, :, None]
        )
        mask_padding[:, max_points:] = (
            torch.arange(max_points, dtype=torch.long, device=X.device).expand(
                batch_size, -1
            )
            >= num_points[1, :, None]
        )

    C = torch.empty((batch_size, ksamples, emb_size), device=X.device)

    batch_idx = torch.arange(batch_size, dtype=torch.long, device=X.device)[:, None]
    mask_points = ~mask_padding
    mask_points[mask_points.sum(-1) == 0, :] = True
    C[batch_idx, 0] = X[
        batch_idx, torch.multinomial(mask_points.float(), 1, replacement=True)
    ]

    dist_XC = torch.empty((batch_size, max_points2, ksamples - 1), device=X.device)
    for i in range(1, ksamples):
        add_kmeanspp_sample(X, C, dist_XC, i, mask_padding, dist)
    return C
