import math

import torch
from munch import Munch

from lcn.cost import distances, kernels
from lcn.cost.landmarks import get_landmarks
from lcn.sinkhorn import arg_log_sinkhorn2
from lcn.utils import loginvexp, logsumexp_signed_to_signed, repeat_blocks, scatter


def get_matrix_stack_idx(num_points):
    idx1 = torch.arange(num_points.sum(), device=num_points.device)
    num_points_rpt = torch.repeat_interleave(num_points, num_points)
    idx1 = torch.repeat_interleave(idx1, num_points_rpt)
    idx2 = repeat_blocks(num_points, num_points)
    return torch.stack((idx1, idx2))


def calc_cost_matrix(embeddings, num_points, dist):
    _, max_points, _ = embeddings[0].shape

    # Real size: b x n1 x n2
    cost_matrix = dist.cdist(embeddings[0], embeddings[1])

    # This mask gives everything that is not a real calculated distance
    mask_n1 = (
        torch.arange(max_points, dtype=torch.int64, device=num_points.device)[
            :, None
        ].expand_as(cost_matrix)
        >= num_points[0, :, None, None]
    )
    mask_n2 = (
        torch.arange(max_points, dtype=torch.int64, device=num_points.device).expand_as(
            cost_matrix
        )
        >= num_points[1, :, None, None]
    )
    mask_nodist = mask_n1 | mask_n2

    # Set all non-distances to inf
    cost_matrix = cost_matrix.masked_fill(mask_nodist, math.inf)

    return Munch(cost_mat=cost_matrix, num_points=num_points)


def calc_bp_cost_matrix(embeddings, num_points, dist, alpha):
    batch_size, max_points, _ = embeddings[0].shape

    # Inner size: b x n1 x n2
    cost_matrix = dist.cdist(embeddings[0], embeddings[1])

    # This mask gives everything that is a real calculated distance
    range_points = torch.arange(max_points, dtype=torch.int64, device=num_points.device)
    mask_n1 = (
        range_points[:, None].expand_as(cost_matrix) >= num_points[0, :, None, None]
    )
    mask_n2 = range_points.expand_as(cost_matrix) >= num_points[1, :, None, None]
    mask_nodist = mask_n1 | mask_n2

    # Set all non-distances to inf
    cost_matrix = cost_matrix.masked_fill(mask_nodist, math.inf)

    # Norms of embeddings (scaled by learnable alpha)
    if alpha.shape[0] == 1:
        norms1 = alpha ** 2 * dist.norm(embeddings[0])
        norms2 = alpha ** 2 * dist.norm(embeddings[1])
    else:
        norms1 = dist.norm(alpha[None, :] * embeddings[0])
        norms2 = dist.norm(alpha[None, :] * embeddings[1])

    # Fill non-norms with zeros, then with very large number (infinity causes backprop errors)
    mask_norms_n1 = range_points.expand_as(norms1) >= num_points[0, :, None]
    mask_norms_n2 = range_points.expand_as(norms2) >= num_points[1, :, None]
    norms1 = norms1.masked_fill(mask_norms_n1, 1e20)
    norms2 = norms2.masked_fill(mask_norms_n2, 1e20)

    return Munch(
        cost_mat=cost_matrix, norms1=norms1, norms2=norms2, num_points=num_points
    )


def calc_bp_cost_matrix_full(embeddings, num_points, dist, alpha):
    batch_size, max_points, _ = embeddings[0].shape

    # Inner size: b x n1 x n2
    max_n1n2 = num_points.sum(0).max()
    cost_matrix = embeddings[0].new_zeros(batch_size, max_n1n2, max_n1n2)
    cost_matrix[:, :max_points, :max_points] = dist.cdist(embeddings[0], embeddings[1])

    # This mask gives everything that is a real calculated distance
    range_points = torch.arange(max_n1n2, dtype=torch.int64, device=num_points.device)
    mask_in_n1 = (
        range_points[:, None].expand_as(cost_matrix) < num_points[0, :, None, None]
    )
    mask_in_n2 = range_points.expand_as(cost_matrix) < num_points[1, :, None, None]
    mask_dist = mask_in_n1 & mask_in_n2

    # Set all non-distances to 0
    cost_matrix = cost_matrix.masked_fill(~mask_dist, 0)

    # This mask gives the matrix needed for matching (without padding)
    num_points_sum = num_points.sum(0)
    mask_inner1 = (
        range_points[:, None].expand_as(cost_matrix) < num_points_sum[:, None, None]
    )
    mask_inner2 = range_points.expand_as(cost_matrix) < num_points_sum[:, None, None]
    mask_inner = mask_inner1 & mask_inner2

    # Fill non-distance values in inner matrix with norms of embeddings (scaled by learnable alpha)
    if alpha.shape[0] == 1:
        fill_val1 = alpha ** 2 * dist.norm(embeddings[0])
        fill_val2 = alpha ** 2 * dist.norm(embeddings[1])
    else:
        fill_val1 = dist.norm(alpha[None, :] * embeddings[0])
        fill_val2 = dist.norm(alpha[None, :] * embeddings[1])

    mask_diag1 = range_points.expand_as(cost_matrix) == num_points[1][
        :, None, None
    ] + range_points[:, None].expand_as(cost_matrix)
    mask_diag2 = range_points[:, None].expand_as(cost_matrix) == num_points[0][
        :, None, None
    ] + range_points.expand_as(cost_matrix)
    cost_matrix[:, :max_points, :] += (
        fill_val1[:, :, None]
        * (mask_inner & mask_in_n1 & ~mask_in_n2 & mask_diag1)[:, :max_points, :]
    )
    cost_matrix[:, :max_points, :].masked_fill_(
        (mask_inner & mask_in_n1 & ~mask_in_n2 & ~mask_diag1)[:, :max_points, :],
        math.inf,
    )
    cost_matrix[:, :, :max_points] += (
        fill_val2[:, None, :]
        * (mask_inner & ~mask_in_n1 & mask_in_n2 & mask_diag2)[:, :, :max_points]
    )
    cost_matrix[:, :, :max_points].masked_fill_(
        (mask_inner & ~mask_in_n1 & mask_in_n2 & ~mask_diag2)[:, :, :max_points],
        math.inf,
    )

    return Munch(cost_mat=cost_matrix, num_points=num_points)


@torch.no_grad()
def get_pair_indices_matched_clusters(
    num_points,
    landmarks,
    num_clusters,
    cluster,
    sinkhorn_reg,
    dist,
    sinkhorn_niter,
    num_hashes,
    multiscale,
    threshold,
):
    batch_size, max_points = cluster[0].shape

    # Get cluster sizes
    cluster_sizes1 = scatter(
        cluster[0].new_ones(1).expand_as(cluster[0]),
        cluster[0],
        dim=1,
        dim_size=num_clusters + 1,
        reduce="sum",
    )[:, :num_clusters].flatten()
    cluster_sizes2 = scatter(
        cluster[1].new_ones(1).expand_as(cluster[1]),
        cluster[1],
        dim=1,
        dim_size=num_clusters + 1,
        reduce="sum",
    )[:, :num_clusters].flatten()

    # Batch offsets for the relative indices
    offsets_left = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=num_points.device),
            num_points[0, :-1].cumsum(0),
        )
    )
    offsets_right = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=num_points.device),
            num_points[1, :-1].cumsum(0),
        )
    )

    # Get points in each cluster
    batch_idx = torch.arange(
        batch_size, dtype=cluster[0].dtype, device=num_points.device
    )[:, None].expand(-1, max_points)
    cluster_point_left_all = torch.argsort(cluster[0])
    cluster_point_left = cluster_point_left_all[
        (cluster[0] < num_clusters)[batch_idx, cluster_point_left_all]
    ]
    cluster_point_right_all = torch.argsort(cluster[1])
    cluster_point_right = cluster_point_right_all[
        (cluster[1] < num_clusters)[batch_idx, cluster_point_right_all]
    ]

    if multiscale:
        assert num_hashes == 1

        # Coarse scale: Matching between landmarks
        if isinstance(dist, kernels.Kernel):
            dist = distances.KernelDist(dist)
        dist_lm = dist.cdist(landmarks[0], landmarks[1])

        # Match landmarks
        dist_lm_len = dist_lm.new_tensor([num_clusters])
        T_log_lm, _, _ = arg_log_sinkhorn2(
            dist_lm, dist_lm_len, sinkhorn_reg, sinkhorn_niter
        )
        T_lm = T_log_lm.exp()

        choice = T_lm > threshold
        assert (
            choice.any()
        ), f"No options above threshold {threshold}. maximum = {T_lm.max().cpu()}"

        # Indices for fine scale
        # This can probably be done much simpler via argsort.

        # Cluster pairs
        cl_pairs_batch, cl_pairs_left, cl_pairs_right = torch.where(choice)
        cl_pairs_offset_left = num_clusters * cl_pairs_batch + cl_pairs_left
        cl_pairs_offset_right = num_clusters * cl_pairs_batch + cl_pairs_right

        # Number of right points in each cluster pair
        cl_pair_sizes2 = cluster_sizes2[cl_pairs_offset_right]

        # Number of copies per cluster
        cl_reps_left = scatter(
            cl_pair_sizes2,
            cl_pairs_offset_left,
            dim_size=batch_size * num_clusters,
            reduce="sum",
        )
        point_reps_left = torch.repeat_interleave(cl_reps_left, cluster_sizes1)

        # Left indices of point pairs
        dists_idx_nooffset_left = torch.repeat_interleave(
            cluster_point_left, point_reps_left
        )

        # Relative indices of right points in each cluster pair
        # We construct these by misusing repeat_blocks to give indices of present cluster pairs
        # and offset them by the number of points in non-present pairs.
        cluster_sizes2_expanded = (
            cluster_sizes2.reshape(batch_size, 1, num_clusters)
            .expand(batch_size, num_clusters, num_clusters)
            .flatten()
        )
        points_right_cl_idx_raw = repeat_blocks(
            cluster_sizes2_expanded, choice.flatten().long()
        )

        # Point offsets for the relative right point indices
        point_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=num_points.device),
                num_points[1, :, None]
                .expand(batch_size, num_clusters)
                .flatten()[:-1]
                .cumsum(0),
            )
        )

        # Relative indices of right points with correct offset
        offsets_right_cl_idx = (
            offsets_right[:, None].expand(batch_size, num_clusters).flatten()
            - point_offsets
        )
        offsets_right_cl_idx_repeated = torch.repeat_interleave(
            offsets_right_cl_idx, cl_reps_left
        )
        points_right_cl_idx = points_right_cl_idx_raw + offsets_right_cl_idx_repeated

        # Indices of right points in each cluster pair
        points_right_cl = cluster_point_right[points_right_cl_idx]

        # Right indices of point pairs
        points_right_idx = repeat_blocks(cl_reps_left, cluster_sizes1)
        dists_idx_nooffset_right = points_right_cl[points_right_idx]

        # Batch index of each point pair
        pairs_per_sample = (
            (cl_reps_left * cluster_sizes1).reshape(batch_size, num_clusters).sum(1)
        )
    else:
        cluster_sizes2_rep = torch.repeat_interleave(cluster_sizes2, cluster_sizes1)
        dists_idx_nooffset_left = torch.repeat_interleave(
            cluster_point_left, cluster_sizes2_rep
        )

        points_right_idx = repeat_blocks(cluster_sizes2, cluster_sizes1)
        dists_idx_nooffset_right = cluster_point_right[points_right_idx]

        pairs_per_sample = (
            (cluster_sizes1 * cluster_sizes2).reshape(batch_size, num_clusters).sum(1)
        )

    if num_hashes == 1:
        # Batch index of each point pair
        pairs_batch_idx = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long, device=num_points.device),
            pairs_per_sample,
        )
        # Set up left distance matrix indices
        dists_idx_left = dists_idx_nooffset_left + torch.repeat_interleave(
            offsets_left, pairs_per_sample
        )

        # Sort indices for efficiency
        resort_idx = torch.argsort(dists_idx_left)
        dists_idx_left = dists_idx_left[resort_idx]
        dists_idx_nooffset_left = dists_idx_nooffset_left[resort_idx]
        dists_idx_nooffset_right = dists_idx_nooffset_right[resort_idx]
    else:
        real_batch_size = batch_size // num_hashes

        # Batch index of each point pair
        batch_idx = torch.repeat_interleave(
            torch.arange(real_batch_size, dtype=torch.long, device=num_points.device),
            num_hashes,
        )
        pairs_batch_idx = torch.repeat_interleave(batch_idx, pairs_per_sample)

        pairs = torch.stack(
            (pairs_batch_idx, dists_idx_nooffset_left, dists_idx_nooffset_right), dim=1
        )
        pairs_unique = torch.unique(pairs, sorted=True, dim=0)
        (
            pairs_batch_idx,
            dists_idx_nooffset_left,
            dists_idx_nooffset_right,
        ) = pairs_unique.unbind(dim=1)

        pairs_per_sample = torch.bincount(pairs_batch_idx, minlength=real_batch_size)

        # Set up left distance matrix indices
        dists_idx_left = dists_idx_nooffset_left + torch.repeat_interleave(
            offsets_left, pairs_per_sample
        )

    # Set up right distance matrix indices
    dists_idx_right = dists_idx_nooffset_right + torch.repeat_interleave(
        offsets_right, pairs_per_sample
    )

    return (
        pairs_batch_idx,
        dists_idx_nooffset_left,
        dists_idx_nooffset_right,
        dists_idx_left,
        dists_idx_right,
    )


def calc_cost_matrix_sparse(
    embeddings,
    num_points,
    landmarks,
    num_clusters,
    cluster,
    sinkhorn_reg,
    dist,
    alpha,
    sinkhorn_niter,
    num_hashes,
    multiscale,
    threshold,
    calc_norms=True,
):
    batch_size, _, _ = embeddings[0].shape

    (
        pairs_batch_idx,
        dists_idx_nooffset_left,
        dists_idx_nooffset_right,
        dists_idx_left,
        dists_idx_right,
    ) = get_pair_indices_matched_clusters(
        num_points,
        landmarks,
        num_clusters,
        cluster,
        sinkhorn_reg,
        dist,
        sinkhorn_niter,
        num_hashes,
        multiscale,
        threshold,
    )

    # Fine scale
    dists = dist.pairwise_distance(
        embeddings[0][pairs_batch_idx, dists_idx_nooffset_left],
        embeddings[1][pairs_batch_idx, dists_idx_nooffset_right],
    )

    # Compute norm indices
    norms1_batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=num_points.device),
        num_points[0],
    )
    norms1_idx = repeat_blocks(num_points[0], 1, continuous_indexing=False)
    norms2_batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.long, device=num_points.device),
        num_points[1],
    )
    norms2_idx = repeat_blocks(num_points[1], 1, continuous_indexing=False)

    if calc_norms:
        # Calculate norms
        norms1 = embeddings[0][norms1_batch_idx, norms1_idx]
        norms2 = embeddings[1][norms2_batch_idx, norms2_idx]
        if alpha.shape[0] == 1:
            norms1 = alpha ** 2 * dist.norm(norms1)
            norms2 = alpha ** 2 * dist.norm(norms2)
        else:
            norms1 = dist.norm(alpha[None, :] * norms1)
            norms2 = dist.norm(alpha[None, :] * norms2)
    else:
        norms1 = None
        norms2 = None

    return Munch(
        costs=dists,
        cost_batch_idx=pairs_batch_idx,
        cost_idx1=dists_idx_left,
        cost_idx2=dists_idx_right,
        cost_idx1_nooffset=dists_idx_nooffset_left,
        cost_idx2_nooffset=dists_idx_nooffset_right,
        norms1_batch_idx=norms1_batch_idx,
        norms1_idx=norms1_idx,
        norms1=norms1,
        norms2_batch_idx=norms2_batch_idx,
        norms2_idx=norms2_idx,
        norms2=norms2,
        num_points=num_points,
        sinkhorn_reg=sinkhorn_reg,
    )


def calc_cost_matrix_nystrom(
    embeddings, num_points, landmarks, sinkhorn_reg, dist, alpha, calc_norms=True
):
    batch_size, max_points, emb_size = embeddings[0].shape

    # Norms of embeddings (scaled by learnable alpha)
    if calc_norms:
        if alpha.shape[0] == 1:
            norms1 = alpha ** 2 * dist.norm(embeddings[0])
            norms2 = alpha ** 2 * dist.norm(embeddings[1])
        else:
            norms1 = dist.norm(alpha[None, :] * embeddings[0])
            norms2 = dist.norm(alpha[None, :] * embeddings[1])

        # Fill non-norms with zeros, then with very large number (infinity causes backprop errors)
        mask_n1 = (
            torch.arange(
                max_points, dtype=torch.int64, device=num_points.device
            ).expand_as(norms1)
            >= num_points[0, :, None]
        )
        mask_n2 = (
            torch.arange(
                max_points, dtype=torch.int64, device=num_points.device
            ).expand_as(norms2)
            >= num_points[1, :, None]
        )
        norms1 = norms1.masked_fill(mask_n1, 1e20)
        norms2 = norms2.masked_fill(mask_n2, 1e20)
    else:
        norms1 = norms2 = None

    dist_1a = dist.cdist(embeddings[0], landmarks)
    dist_aa = dist.cdist(landmarks, landmarks)
    dist_a2 = dist.cdist(landmarks, embeddings[1])

    dist_aa = dist_aa / sinkhorn_reg[:, None, None]

    sim_aa_inv, sign_aa = loginvexp(-dist_aa, use_double=True)

    sim_a2_scaled, sign_a2 = logsumexp_signed_to_signed(
        sim_aa_inv[:, :, :, None]
        - (dist_a2 / sinkhorn_reg[:, None, None])[:, None, :, :],
        sign_aa[:, :, :, None],
        dim=2,
    )

    # This mask gives everything that is not a real calculated distance
    mask_dist_1a = (
        torch.arange(max_points, dtype=torch.int64, device=num_points.device)[
            :, None
        ].expand_as(dist_1a)
        >= num_points[0, :, None, None]
    )
    mask_sim_a2 = (
        torch.arange(max_points, dtype=torch.int64, device=num_points.device).expand_as(
            dist_a2
        )
        >= num_points[1, :, None, None]
    )

    # Set all non-distances to very large negative number (infinity causes backprop errors)
    dist_1a = dist_1a.masked_fill(mask_dist_1a, 1e20)
    sim_a2_scaled = sim_a2_scaled.masked_fill(mask_sim_a2, -1e20)
    sign_a2 = sign_a2.masked_fill(mask_sim_a2, 1)

    return Munch(
        cost_1a=dist_1a,
        sim_a2_scaled=sim_a2_scaled,
        sign_a2=sign_a2,
        norms1=norms1,
        norms2=norms2,
        num_points=num_points,
        sinkhorn_reg=sinkhorn_reg,
    )


def merge_cost_matrices(cost_mat_nystrom, cost_mat_sparse):
    cost_batch_idx = cost_mat_sparse.cost_batch_idx
    cost_idx1_nooffset = cost_mat_sparse.cost_idx1_nooffset
    cost_idx2_nooffset = cost_mat_sparse.cost_idx2_nooffset

    # This is super-linear if Nyström and sparse use the same landmarks/clusters.
    # l1 * n * n/l2, l1 constant, l2~n^(2/3) -> l1 * n^(4/3)
    # => use different clusters for Nyström and sparse
    sim_approx_scaled, sign_approx = logsumexp_signed_to_signed(
        cost_mat_nystrom.sim_a2_scaled[cost_batch_idx, :, cost_idx2_nooffset]
        - (cost_mat_nystrom.cost_1a / cost_mat_nystrom.sinkhorn_reg[:, None, None])[
            cost_batch_idx, cost_idx1_nooffset, :
        ],
        cost_mat_nystrom.sign_a2[cost_batch_idx, :, cost_idx2_nooffset],
        dim=-1,
    )

    cost_mat = cost_mat_nystrom.copy()
    cost_mat.update(
        cost_exact=cost_mat_sparse.costs,
        sim_approx_scaled=sim_approx_scaled,
        sign_approx=sign_approx,
        corr_batch_idx=cost_batch_idx,
        corr_idx1=cost_mat_sparse.cost_idx1,
        corr_idx2=cost_mat_sparse.cost_idx2,
        corr_idx1_nooffset=cost_idx1_nooffset,
        corr_idx2_nooffset=cost_idx2_nooffset,
        norms1_batch_idx=cost_mat_sparse.norms1_batch_idx,
        norms1_idx=cost_mat_sparse.norms1_idx,
        norms2_batch_idx=cost_mat_sparse.norms2_batch_idx,
        norms2_idx=cost_mat_sparse.norms2_idx,
    )
    return cost_mat


def get_cost_matrix(
    embeddings,
    num_points,
    nystrom,
    sparse,
    sinkhorn_reg,
    sinkhorn_niter,
    alpha,
    dist,
    dist_cluster=None,
    bp_cost_matrix=True,
    full_bp_matrix=False,
):
    batch_size, max_points, emb_size = embeddings[0].shape

    if not dist_cluster:
        dist_cluster = dist

    # Here 2 embeddings tensors with batch_size x max_num_points x emb_size
    if sparse:
        assert (
            not full_bp_matrix
        ), "Full BP matrix only usable for full (non-approximate) Sinkhorn."
        if sparse["method"] == "multiscale":
            separate_landmarks = True
            assert sparse.get("num_hash_bands", 1) == 1
            assert sparse.get("num_hashes_per_band", 1) == 1
        else:
            separate_landmarks = False
        landmarks_sparse, cluster = get_landmarks(
            sparse["neighbor_method"],
            embeddings,
            num_points,
            sparse["num_clusters"],
            dist=dist_cluster,
            num_hashbands=sparse.get("num_hash_bands", 1),
            num_hashes_per_band=sparse.get("num_hashes_per_band", 1),
            return_assignments=True,
            separate=separate_landmarks,
        )
        if isinstance(sparse["num_clusters"], list):
            num_clusters_flat = torch.prod(
                torch.tensor(sparse["num_clusters"])
                ** sparse.get("num_hashes_per_band", 1)
            )
        else:
            num_clusters_flat = sparse["num_clusters"] ** sparse.get(
                "num_hashes_per_band", 1
            )
        if separate_landmarks:
            landmarks_sparse = landmarks_sparse.reshape(
                2, -1, num_clusters_flat, emb_size
            )
        cluster = [cl.reshape(-1, max_points) for cl in cluster]
        cost_matrix_sparse = calc_cost_matrix_sparse(
            embeddings,
            num_points,
            landmarks_sparse,
            num_clusters_flat,
            cluster,
            sinkhorn_reg=sinkhorn_reg,
            dist=dist,
            alpha=alpha,
            sinkhorn_niter=sinkhorn_niter,
            num_hashes=sparse.get("num_hash_bands", 1),
            multiscale=(sparse["method"] == "multiscale"),
            threshold=sparse.get("multiscale_threshold", 0),
            calc_norms=(nystrom is None and bp_cost_matrix),
        )

    if nystrom:
        assert (
            not full_bp_matrix
        ), "Full BP matrix only usable for full (non-approximate) Sinkhorn."
        if (
            sparse
            and (nystrom["landmark_method"] == sparse["neighbor_method"])
            and (nystrom["num_clusters"] == sparse["num_clusters"])
            and (sparse.get("num_hash_bands", 1) == 1)
            and (sparse.get("num_hashes_per_band", 1) == 1)
        ):
            landmarks_nystrom = landmarks_sparse
        else:
            landmarks_nystrom = get_landmarks(
                nystrom["landmark_method"],
                embeddings,
                num_points,
                nystrom["num_clusters"],
                dist=dist_cluster,
                num_hashbands=1,
                num_hashes_per_band=1,
                return_assignments=False,
            )
            if isinstance(nystrom["num_clusters"], list):
                num_clusters_flat = torch.prod(torch.tensor(nystrom["num_clusters"]))
            else:
                num_clusters_flat = nystrom["num_clusters"]
        landmarks_nystrom = landmarks_nystrom.reshape(
            batch_size, num_clusters_flat, emb_size
        )
        cost_matrix_nys = calc_cost_matrix_nystrom(
            embeddings,
            num_points,
            landmarks_nystrom,
            sinkhorn_reg=sinkhorn_reg,
            dist=dist,
            alpha=alpha,
            calc_norms=bp_cost_matrix,
        )

    if sparse and nystrom:
        return merge_cost_matrices(cost_matrix_nys, cost_matrix_sparse)
    elif sparse:
        return cost_matrix_sparse
    elif nystrom:
        return cost_matrix_nys
    elif bp_cost_matrix:
        if full_bp_matrix:
            cost_matrix = calc_bp_cost_matrix_full(embeddings, num_points, dist, alpha)
        else:
            cost_matrix = calc_bp_cost_matrix(embeddings, num_points, dist, alpha)
        cost_matrix.sinkhorn_reg = sinkhorn_reg
        return cost_matrix
    else:
        cost_matrix = calc_cost_matrix(embeddings, num_points, dist)
        cost_matrix.sinkhorn_reg = sinkhorn_reg
        return cost_matrix
