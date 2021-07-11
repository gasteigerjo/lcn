import torch

from lcn.utils import scatter, segment_coo


# For some reason scripting this is broken in PyTorch>=1.8
# (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
@torch.jit.script
def sparse_lse_uv(sim_mat, cost_idx1, cost_idx2, norms_batch_idx, vec, dim: int):
    if dim == 1:
        sum_inner = scatter(
            sim_mat + vec[cost_idx1],
            cost_idx2,
            dim_size=norms_batch_idx.shape[0],
            reduce="logsumexp",
        )
        return sum_inner
    else:
        mat_inner = sim_mat + vec[cost_idx2]
        lse_offset = segment_coo(
            mat_inner, cost_idx1, dim_size=norms_batch_idx.shape[0], reduce="max"
        )
        sum_inner = segment_coo(
            (mat_inner - lse_offset[cost_idx1]).exp(),
            cost_idx1,
            dim_size=norms_batch_idx.shape[0],
            reduce="sum",
        )
        return torch.log(sum_inner) + lse_offset


def arg_log_sparse_sinkhorn(
    costs: torch.FloatTensor,
    cost_batch_idx: torch.LongTensor,
    cost_idx1: torch.LongTensor,
    cost_idx2: torch.LongTensor,
    norms1_batch_idx: torch.LongTensor,
    norms2_batch_idx: torch.LongTensor,
    num_points: torch.LongTensor,
    sinkhorn_reg: torch.FloatTensor,
    niter: int = 50,
):
    sim_scaled = -costs / sinkhorn_reg[cost_batch_idx]

    u = costs.new_zeros(num_points[0].sum())
    v = costs.new_zeros(num_points[1].sum())
    for _ in range(niter):
        u = -sparse_lse_uv(sim_scaled, cost_idx1, cost_idx2, norms1_batch_idx, v, dim=2)
        v = -sparse_lse_uv(sim_scaled, cost_idx1, cost_idx2, norms2_batch_idx, u, dim=1)

    T_log = sim_scaled + u[cost_idx1] + v[cost_idx2]

    return T_log, u, v


class LogSparseSinkhorn(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using a sparse cost matrix. Calculated in log space.
    """

    @staticmethod
    def forward(
        ctx,
        costs: torch.FloatTensor,
        cost_batch_idx: torch.LongTensor,
        cost_idx1: torch.LongTensor,
        cost_idx2: torch.LongTensor,
        norms1_batch_idx: torch.LongTensor,
        norms2_batch_idx: torch.LongTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size = num_points.shape[1]

        T_log, u, v = arg_log_sparse_sinkhorn(
            costs,
            cost_batch_idx,
            cost_idx1,
            cost_idx2,
            norms1_batch_idx,
            norms2_batch_idx,
            num_points,
            sinkhorn_reg,
            niter,
        )

        T = T_log.exp()

        ctx.save_for_backward(T, cost_batch_idx)

        C = sinkhorn_reg * segment_coo(
            (u[cost_idx1] + v[cost_idx2]) * T,
            cost_batch_idx,
            dim_size=batch_size,
            reduce="sum",
        )

        return C

    @staticmethod
    def backward(ctx, grad_output):
        T, cost_batch_idx = ctx.saved_tensors

        return (
            T * grad_output[cost_batch_idx],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LogSparseSinkhornBP(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using the BP matrix for unbalanced distributions
    and a sparse cost matrix. Calculated in log space.
    """

    # For some reason scripting this is broken in PyTorch>=1.8
    # (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
    @staticmethod
    @torch.jit.script
    def _lse_uv_inner(
        costs,
        cost_vec_idx,
        cost_scatter_idx,
        norms,
        norms_outer_batch_idx,
        vec,
        batch_size: int,
    ):
        outer_start = vec.shape[0] - norms.shape[0]

        mat_inner = -costs + vec[cost_vec_idx]

        max_inner = scatter(
            mat_inner,
            cost_scatter_idx,
            dim_size=norms_outer_batch_idx.shape[0],
            reduce="max",
        )
        max_outer = vec[outer_start:] - norms
        lse_offset = torch.max(max_inner, max_outer)

        sum_inner = scatter(
            (mat_inner - lse_offset[cost_scatter_idx]).exp(),
            cost_scatter_idx,
            dim_size=norms_outer_batch_idx.shape[0],
            reduce="sum",
        )

        sum_outer = torch.exp(-norms - lse_offset + vec[outer_start:])

        return torch.log(sum_inner + sum_outer) + lse_offset

    @staticmethod
    @torch.jit.script
    def _lse_uv_outer(norms, inner_batch_idx, outer_batch_idx, vec, batch_size: int):
        outer_start = inner_batch_idx.shape[0]

        inner_part = -norms + vec[:outer_start]
        logsum_outer = scatter(
            vec[outer_start:], outer_batch_idx, dim_size=batch_size, reduce="logsumexp"
        )

        lse_offset = torch.max(inner_part, logsum_outer[inner_batch_idx])

        sum_inner = (inner_part - lse_offset).exp()
        sum_outer = (logsum_outer[inner_batch_idx] - lse_offset).exp()

        return torch.log(sum_inner + sum_outer) + lse_offset[inner_batch_idx]

    @staticmethod
    def forward(
        ctx,
        costs: torch.FloatTensor,
        cost_batch_idx: torch.LongTensor,
        cost_idx1: torch.LongTensor,
        cost_idx2: torch.LongTensor,
        norms1_batch_idx: torch.LongTensor,
        norms1: torch.FloatTensor,
        norms2_batch_idx: torch.LongTensor,
        norms2: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size = num_points.shape[1]
        num_points_sums = num_points.sum(1)

        cost_scaled = costs / sinkhorn_reg[cost_batch_idx]
        norms1_scaled = norms1 / sinkhorn_reg[norms1_batch_idx]
        norms2_scaled = norms2 / sinkhorn_reg[norms2_batch_idx]

        u = costs.new_zeros(num_points.sum())
        v = costs.new_zeros(num_points.sum())
        for _ in range(niter):
            u[: num_points_sums[0]] = -LogSparseSinkhornBP._lse_uv_inner(
                cost_scaled,
                cost_idx2,
                cost_idx1,
                norms1_scaled,
                norms1_batch_idx,
                v,
                batch_size,
            )
            u[num_points_sums[0] :] = -LogSparseSinkhornBP._lse_uv_outer(
                norms2_scaled, norms2_batch_idx, norms1_batch_idx, v, batch_size
            )
            v[: num_points_sums[1]] = -LogSparseSinkhornBP._lse_uv_inner(
                cost_scaled,
                cost_idx1,
                cost_idx2,
                norms2_scaled,
                norms2_batch_idx,
                u,
                batch_size,
            )
            v[num_points_sums[1] :] = -LogSparseSinkhornBP._lse_uv_outer(
                norms1_scaled, norms1_batch_idx, norms2_batch_idx, u, batch_size
            )

        T11 = (-cost_scaled + u[cost_idx1] + v[cost_idx2]).exp()
        T12 = torch.exp(
            -norms1_scaled + u[: num_points_sums[0]] + v[num_points_sums[1] :]
        )
        T21 = torch.exp(
            -norms2_scaled + v[: num_points_sums[1]] + u[num_points_sums[0] :]
        )

        ctx.save_for_backward(
            T11, T12, T21, cost_batch_idx, norms1_batch_idx, norms2_batch_idx
        )

        T22_u = torch.exp(
            u[num_points_sums[0] :]
            + scatter(
                v[num_points_sums[1] :],
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="logsumexp",
            )[norms2_batch_idx]
        )
        T22_v = torch.exp(
            scatter(
                u[num_points_sums[0] :],
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="logsumexp",
            )[norms1_batch_idx]
            + v[num_points_sums[1] :]
        )

        C11 = sinkhorn_reg * segment_coo(
            (u[cost_idx1] + v[cost_idx2]) * T11,
            cost_batch_idx,
            dim_size=batch_size,
            reduce="sum",
        )
        C12 = sinkhorn_reg * (
            segment_coo(
                u[: num_points_sums[0]] * T12,
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
            + segment_coo(
                v[num_points_sums[1] :] * T12,
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
        )
        C21 = sinkhorn_reg * (
            segment_coo(
                u[num_points_sums[0] :] * T21,
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
            + segment_coo(
                v[: num_points_sums[1]] * T21,
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
        )
        C22 = sinkhorn_reg * (
            segment_coo(
                u[num_points_sums[0] :] * T22_u,
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
            + segment_coo(
                v[num_points_sums[1] :] * T22_v,
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
        )
        C = C11 + C12 + C21 + C22

        return C

    @staticmethod
    def backward(ctx, grad_output):
        (
            T11,
            T12,
            T21,
            cost_batch_idx,
            norms1_batch_idx,
            norms2_batch_idx,
        ) = ctx.saved_tensors

        return (
            T11 * grad_output[cost_batch_idx],
            None,
            None,
            None,
            None,
            T12 * grad_output[norms1_batch_idx],
            None,
            T21 * grad_output[norms2_batch_idx],
            None,
            None,
            None,
        )
