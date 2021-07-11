import torch

from lcn.cost import distances
from lcn.cost.sampling import get_kmeanspp_sample, kmeanspp_sampling_cat
from lcn.utils import get_matching_embeddings, repeat_blocks, scatter


@torch.no_grad()
def get_cluster_assignments(
    X: torch.Tensor,
    C: torch.Tensor,
    mask_value: int,
    dist: distances.Distance,
    num_points: torch.Tensor = None,
    mask: torch.Tensor = None,
):
    batch_size, max_points, emb_size = X.shape
    if C.dim() == 4:
        num_hashes = C.shape[1]
    else:
        num_hashes = 1

    if mask is None:
        mask = (
            torch.arange(max_points, dtype=torch.long, device=X.device).expand(
                batch_size, num_hashes, -1
            )
            >= num_points[:, None, None]
        )

    C = C.reshape(batch_size, -1, emb_size)
    D_ij = dist.cdist(X, C)  # (batch_size, 2*max_points, num_hashes*kclusters)
    D_ij = D_ij.reshape(batch_size, max_points, num_hashes, -1)
    cluster = D_ij.argmin(
        dim=-1
    ).long()  # (batch_size, 2*max_points, num_hashes) Nearest cluster
    if num_hashes == 1:
        cluster = cluster.squeeze(-1)  # (batch_size, 2*max_points)
    else:
        cluster = cluster.transpose(1, 2)  # (batch_size, num_hashes, 2*max_points)
    cluster.masked_fill_(mask, mask_value)
    return cluster


@torch.no_grad()
def kmeans_cat(X, num_points, kclusters, niter, dist, kmeanspp_init=False):
    batch_size, max_points2, _ = X.shape
    max_points = max_points2 // 2
    assert max_points >= kclusters
    assert num_points.dim() == 2

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

    return kmeans_masked(
        X, mask_padding, kclusters, niter, dist, kmeanspp_init=kmeanspp_init
    )


@torch.no_grad()
def kmeans(X, num_points, kclusters, niter, dist, kmeanspp_init=False):
    batch_size, max_points, _ = X.shape
    assert max_points >= kclusters
    assert num_points.dim() == 1

    mask_padding = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, -1
        )
        >= num_points[:, None]
    )

    return kmeans_masked(
        X, mask_padding, kclusters, niter, dist, kmeanspp_init=kmeanspp_init
    )


@torch.no_grad()
def kmeans_masked(
    X, mask, kclusters, niter, dist, kmeanspp_init=False, empty_threshold=None
):
    batch_size = X.shape[0]
    if empty_threshold is None:
        empty_threshold = kclusters

    if kmeanspp_init:
        C = kmeanspp_sampling_cat(X, kclusters, dist, mask_padding=mask)
    else:
        # Simple random initialization
        point_idx = torch.multinomial((~mask).float(), kclusters, replacement=False)
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=X.device)[
            :, None
        ].expand_as(point_idx)
        C = X[batch_idx, point_idx]

        # Resample to prevent matching centroids
        batch_matching, idx_matching1, _ = get_matching_embeddings(C, dist=dist)
        while len(batch_matching) > 0:
            C[batch_matching, idx_matching1] = get_kmeanspp_sample(
                X[batch_matching], C[batch_matching], mask[batch_matching], dist=dist
            )
            batch_matching, idx_matching1, _ = get_matching_embeddings(C, dist=dist)

    for i in range(niter):
        cluster = get_cluster_assignments(
            X, C, mask_value=kclusters, dist=dist, mask=mask
        )
        Ncl = scatter(
            X.new_ones(1).expand_as(cluster),
            cluster,
            dim=1,
            dim_size=kclusters + 1,
            reduce="sum",
        )[
            :, :kclusters
        ]  # (batch_size, kclusters) Class weights
        empty_clusters = (Ncl == 0.0) & (Ncl.sum(-1) > empty_threshold)[:, None]

        # Reassign centroids of empty clusters
        if torch.any(empty_clusters):
            batch, C_idx = torch.where(empty_clusters)
            C[batch, C_idx] = get_kmeanspp_sample(
                X[batch], C[batch], mask[batch], dist=dist
            )

            cluster = get_cluster_assignments(
                X, C, mask_value=kclusters, dist=dist, mask=mask
            )
            Ncl = scatter(
                X.new_ones(1).expand_as(cluster),
                cluster,
                dim=1,
                dim_size=kclusters + 1,
                reduce="sum",
            )[
                :, :kclusters
            ]  # (batch_size, kclusters) Class weights

        C = scatter(
            X, cluster[:, :, None], dim=1, dim_size=kclusters + 1, reduce="sum"
        )[:, :kclusters, :] / (Ncl[:, :, None] + 1e-10)

    return C, cluster


def kmeans_hier_inner(
    embeddings_cat, cluster_outer, num_clusters_inner, outer_clusters, niter, dist
):
    batch_size, _, emb_size = embeddings_cat.shape

    cluster_idx, point_idx = torch.sort(cluster_outer, dim=-1)
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=point_idx.device)[
        :, None
    ].expand_as(point_idx)
    non_padding = cluster_idx < outer_clusters
    cluster_idx = (batch_idx * outer_clusters + cluster_idx)[non_padding]
    batch_idx = batch_idx[non_padding]
    point_idx = point_idx[non_padding]

    clusters_total = batch_size * outer_clusters
    cluster_sizes = scatter(
        cluster_idx.new_ones(1).expand_as(cluster_idx),
        cluster_idx,
        dim_size=clusters_total,
        dim=-1,
        reduce="sum",
    )
    max_cluster = cluster_sizes.max()
    mask_padding = torch.zeros(
        (clusters_total, max_cluster), dtype=torch.bool, device=point_idx.device
    )
    mask_padding = (
        torch.arange(max_cluster, dtype=torch.long, device=point_idx.device).expand(
            clusters_total, -1
        )
        >= cluster_sizes[:, None]
    )

    embeddings_inner = embeddings_cat.new_zeros((clusters_total, max_cluster, emb_size))
    point_idx_inner = repeat_blocks(cluster_sizes, 1, continuous_indexing=False)
    embeddings_inner[cluster_idx, point_idx_inner] = embeddings_cat[
        batch_idx, point_idx
    ]

    landmarks, cluster_cat_inner = kmeans_masked(
        embeddings_inner,
        mask_padding,
        kclusters=num_clusters_inner,
        niter=niter,
        dist=dist,
        kmeanspp_init=True,
        empty_threshold=2 * num_clusters_inner,
    )

    cluster_inner = cluster_outer.new_zeros(cluster_outer.shape)
    cluster_inner[batch_idx, point_idx] = cluster_cat_inner[
        cluster_idx, point_idx_inner
    ]

    cluster_cat = cluster_outer * num_clusters_inner + cluster_inner
    return landmarks, cluster_cat
