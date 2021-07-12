import torch

from lcn.lcn_sinkhorn import arg_log_lcn_sinkhorn
from lcn.nystrom_sinkhorn import arg_log_nystrom_sinkhorn
from lcn.sinkhorn import arg_log_sinkhorn2
from lcn.sparse_sinkhorn import arg_log_sparse_sinkhorn
from lcn.utils import call_with_filtered_args, repeat_blocks, scatter


def get_transport_plan(cost_matrix, niter, method="full"):
    """
    Compute transport plan using the provided cost matrix.

    Arguments
    ---------
    cost_matrix: munch.Munch
        Dictionary-like object containing the cost matrix,
        indices, and related information
    niter: int
        Number of Sinkhorn iterations
    method: str
        Which Sinkhorn method to use: full, nystrom, sparse, lcn

    Returns
    -------
    A function that efficiently computes the matrix-vector product
    for a given set of embeddings.
    CAREFUL: The input/output vectors for the resulting function
    need to be properly masked for the batched (padded) version.
    """
    batched = cost_matrix.num_points.shape[1] > 1

    if method == "full":
        T_log, _, _ = call_with_filtered_args(
            arg_log_sinkhorn2, **cost_matrix, niter=niter
        )
        T = T_log.exp()

        if batched:

            def mv_prod(v):
                return torch.einsum("bij,bjk->bik", T, v)

        else:
            T = T.squeeze(0)

            def mv_prod(v):
                return T @ v

    elif method == "nystrom":
        T1a_log, Ta2_log, _, _ = call_with_filtered_args(
            arg_log_nystrom_sinkhorn, **cost_matrix, niter=niter
        )
        T1a, Ta2 = T1a_log.exp(), Ta2_log.exp()

        if batched:

            def mv_prod(v):
                return torch.einsum("bij,bjk,bkl->bil", T1a, Ta2, v)

        else:
            T1a, Ta2 = T1a.squeeze(0), Ta2.squeeze(0)

            def mv_prod(v):
                return T1a @ (Ta2 @ v)

    elif method == "sparse":
        T_log, _, _ = call_with_filtered_args(
            arg_log_sparse_sinkhorn, **cost_matrix, niter=niter
        )
        T = T_log.exp()

        def mv_prod(v):
            return scatter(
                T[:, None] * v[cost_matrix.cost_idx2, :],
                cost_matrix.cost_idx1,
                dim_size=cost_matrix.num_points[0].sum(),
                dim=0,
                reduce="sum",
            )

    elif method == "lcn":
        T1a_log, Ta2_log, T_exact_log, T_approx_log, _, _ = call_with_filtered_args(
            arg_log_lcn_sinkhorn, **cost_matrix, niter=niter
        )
        T1a, Ta2 = T1a_log.exp(), Ta2_log.exp()
        T_delta = T_exact_log.exp() - T_approx_log.exp()

        if batched:
            batch_idx = torch.repeat_interleave(
                torch.arange(
                    cost_matrix.num_points.shape[1],
                    dtype=torch.long,
                    device=T_delta.device,
                ),
                cost_matrix.num_points[0],
            )
            point_idx = repeat_blocks(
                cost_matrix.num_points[0], 1, continuous_indexing=False
            )

            def mv_prod(v):
                corr = scatter(
                    T_delta[:, None]
                    * v[cost_matrix.corr_batch_idx, cost_matrix.corr_idx2_nooffset, :],
                    cost_matrix.corr_idx1,
                    dim_size=cost_matrix.num_points[0].sum(),
                    dim=0,
                    reduce="sum",
                )
                res = torch.einsum("bij,bjk,bkl->bil", T1a, Ta2, v)
                res[batch_idx, point_idx] += corr
                return res

        else:
            T1a, Ta2 = T1a.squeeze(0), Ta2.squeeze(0)

            def mv_prod(v):
                corr = scatter(
                    T_delta[:, None] * v[cost_matrix.corr_idx2, :],
                    cost_matrix.corr_idx1,
                    dim_size=cost_matrix.num_points[0].sum(),
                    dim=0,
                    reduce="sum",
                )
                return T1a @ (Ta2 @ v) + corr

    else:
        raise NotImplementedError(f"Unknown Sinkhorn method '{method}'.")

    return mv_prod
