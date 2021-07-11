import torch

from lcn.utils import logdiffexp, logsumexp_signed_to_signed, scatter, segment_coo


# For some reason scripting this is broken in PyTorch>=1.8
# (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
@torch.jit.script
def lcn_lse_uv(
    sim_1a,
    sim_a2,
    sign_a2,
    sim_corr,
    corr_sign,
    corr_batch_idx,
    corr_idx1,
    corr_idx2,
    corr_idx1_nooffset,
    corr_idx2_nooffset,
    norms1_batch_idx,
    norms1_idx,
    norms2_batch_idx,
    norms2_idx,
    vec,
    dim: int,
):
    if dim % 3 == 2:
        mat_inner1, sign_inner1 = logsumexp_signed_to_signed(
            sim_a2 + vec[:, None, :], sign_a2, dim=2
        )
        mat_inner = sim_1a + mat_inner1[:, None, :]
        sign_inner = sign_inner1[:, None, :]

        lse_offset = mat_inner.max(dim).values
        correction = sim_corr + vec[corr_batch_idx, corr_idx2_nooffset]
        max_corr = segment_coo(
            correction, corr_idx1, dim_size=norms1_batch_idx.shape[0], reduce="max"
        )
        lse_offset[norms1_batch_idx, norms1_idx] = torch.max(
            max_corr, lse_offset[norms1_batch_idx, norms1_idx]
        )

        sum_inner = (sign_inner * torch.exp(mat_inner - lse_offset.unsqueeze(dim))).sum(
            dim
        )
        corr_offset = segment_coo(
            corr_sign
            * torch.exp(correction - lse_offset[corr_batch_idx, corr_idx1_nooffset]),
            corr_idx1,
            dim_size=norms1_batch_idx.shape[0],
            reduce="sum",
        )
        sum_inner[norms1_batch_idx, norms1_idx] += corr_offset

    else:
        mat_inner1 = torch.logsumexp(vec[:, :, None] + sim_1a, dim=1)
        mat_inner = mat_inner1[:, :, None] + sim_a2
        sign_inner = sign_a2

        lse_offset = mat_inner.max(dim).values
        correction = sim_corr + vec[corr_batch_idx, corr_idx1_nooffset]
        max_corr = scatter(
            correction, corr_idx2, dim_size=norms2_batch_idx.shape[0], reduce="max"
        )
        lse_offset[norms2_batch_idx, norms2_idx] = torch.max(
            max_corr, lse_offset[norms2_batch_idx, norms2_idx]
        )

        sum_inner = (sign_inner * torch.exp(mat_inner - lse_offset.unsqueeze(dim))).sum(
            dim
        )
        corr_offset = scatter(
            corr_sign
            * torch.exp(correction - lse_offset[corr_batch_idx, corr_idx2_nooffset]),
            corr_idx2,
            dim_size=norms2_batch_idx.shape[0],
            reduce="sum",
        )
        sum_inner[norms2_batch_idx, norms2_idx] += corr_offset

    return torch.log(sum_inner) + lse_offset


def arg_log_lcn_sinkhorn(
    cost_1a: torch.FloatTensor,
    sim_a2_scaled: torch.FloatTensor,
    sign_a2: torch.FloatTensor,
    cost_exact: torch.FloatTensor,
    sim_approx_scaled: torch.FloatTensor,
    sign_approx: torch.FloatTensor,
    corr_batch_idx: torch.LongTensor,
    corr_idx1: torch.LongTensor,
    corr_idx2: torch.LongTensor,
    corr_idx1_nooffset: torch.LongTensor,
    corr_idx2_nooffset: torch.LongTensor,
    norms1_batch_idx: torch.LongTensor,
    norms1_idx: torch.LongTensor,
    norms2_batch_idx: torch.LongTensor,
    norms2_idx: torch.LongTensor,
    num_points: torch.LongTensor,
    sinkhorn_reg: torch.FloatTensor,
    niter: int = 50,
):
    batch_size, max_points = cost_1a.shape[:2]

    sim_1a_scaled = -cost_1a / sinkhorn_reg[:, None, None]

    sim_exact_scaled = -cost_exact / sinkhorn_reg[0]
    sim_corr, corr_sign = logdiffexp(sim_exact_scaled, sim_approx_scaled, sign_approx)

    u = num_points.new_zeros(batch_size, max_points)
    v = num_points.new_zeros(batch_size, max_points)
    for _ in range(niter):
        u = -lcn_lse_uv(
            sim_1a_scaled,
            sim_a2_scaled,
            sign_a2,
            sim_corr,
            corr_sign,
            corr_batch_idx,
            corr_idx1,
            corr_idx2,
            corr_idx1_nooffset,
            corr_idx2_nooffset,
            norms1_batch_idx,
            norms1_idx,
            norms2_batch_idx,
            norms2_idx,
            v,
            dim=2,
        )
        v = -lcn_lse_uv(
            sim_1a_scaled,
            sim_a2_scaled,
            sign_a2,
            sim_corr,
            corr_sign,
            corr_batch_idx,
            corr_idx1,
            corr_idx2,
            corr_idx1_nooffset,
            corr_idx2_nooffset,
            norms1_batch_idx,
            norms1_idx,
            norms2_batch_idx,
            norms2_idx,
            u,
            dim=1,
        )

    T1a_log = sim_1a_scaled + u[:, :, None]
    Ta2_log = sim_a2_scaled + v[:, None, :]

    T_exact_log = (
        sim_exact_scaled
        + u[corr_batch_idx, corr_idx1_nooffset]
        + v[corr_batch_idx, corr_idx2_nooffset]
    )
    T_approx_log = (
        sim_approx_scaled
        + u[corr_batch_idx, corr_idx1_nooffset]
        + v[corr_batch_idx, corr_idx2_nooffset]
    )

    return T1a_log, Ta2_log, T_exact_log, T_approx_log, u, v


class LogLCNSinkhorn(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using an LCN approximation of the cost matrix.
    Calculated in log space.
    """

    @staticmethod
    def forward(
        ctx,
        cost_1a: torch.FloatTensor,
        sim_a2_scaled: torch.FloatTensor,
        sign_a2: torch.FloatTensor,
        cost_exact: torch.FloatTensor,
        sim_approx_scaled: torch.FloatTensor,
        sign_approx: torch.FloatTensor,
        corr_batch_idx: torch.LongTensor,
        corr_idx1: torch.LongTensor,
        corr_idx2: torch.LongTensor,
        corr_idx1_nooffset: torch.LongTensor,
        corr_idx2_nooffset: torch.LongTensor,
        norms1_batch_idx: torch.LongTensor,
        norms1_idx: torch.LongTensor,
        norms2_batch_idx: torch.LongTensor,
        norms2_idx: torch.LongTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size = num_points.shape[1]

        T1a_log, Ta2_log, T_exact_log, T_approx_log, u, v = arg_log_lcn_sinkhorn(
            cost_1a,
            sim_a2_scaled,
            sign_a2,
            cost_exact,
            sim_approx_scaled,
            sign_approx,
            corr_batch_idx,
            corr_idx1,
            corr_idx2,
            corr_idx1_nooffset,
            corr_idx2_nooffset,
            norms1_batch_idx,
            norms1_idx,
            norms2_batch_idx,
            norms2_idx,
            num_points,
            sinkhorn_reg,
            niter,
        )

        T1a_logsum = torch.logsumexp(T1a_log, dim=1)
        Ta2_logsum, Ta2_sum_sign = logsumexp_signed_to_signed(Ta2_log, sign_a2, dim=2)

        T_sumright = Ta2_sum_sign[:, None, :] * (T1a_log + Ta2_logsum[:, None, :]).exp()
        T_sumleft = sign_a2 * (T1a_logsum[:, :, None] + Ta2_log).exp()

        C_nystrom = sinkhorn_reg * (
            (u * T_sumright.sum(2)).sum(1) + (v * T_sumleft.sum(1)).sum(1)
        )

        T_exact = T_exact_log.exp()
        T_approx = T_approx_log.exp()
        T_delta = T_exact - T_approx

        C_corr = sinkhorn_reg * (
            segment_coo(
                u[norms1_batch_idx, norms1_idx]
                * segment_coo(
                    T_delta, corr_idx1, dim_size=norms1_batch_idx.shape[0], reduce="sum"
                ),
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
            + segment_coo(
                v[norms2_batch_idx, norms2_idx]
                * scatter(
                    T_delta, corr_idx2, dim_size=norms2_batch_idx.shape[0], reduce="sum"
                ),
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
        )

        C = C_nystrom + C_corr

        ctx.save_for_backward(
            T_sumleft, T_sumright, T_exact, T_approx, corr_batch_idx, sinkhorn_reg
        )

        return C

    @staticmethod
    def backward(ctx, grad_output):
        (
            T_sumleft,
            T_sumright,
            T_exact,
            T_approx,
            corr_batch_idx,
            sinkhorn_reg,
        ) = ctx.saved_tensors

        return (
            T_sumright * grad_output[:, None, None],
            -sinkhorn_reg[:, None, None] * T_sumleft * grad_output[:, None, None],
            None,
            T_exact * grad_output[corr_batch_idx],
            sinkhorn_reg[corr_batch_idx] * T_approx * grad_output[corr_batch_idx],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LogLCNSinkhornBP(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using the BP matrix for unbalanced distributions
    and an LCN approximation of the cost matrix.
    Calculated in log space.
    """

    # For some reason scripting this is broken in PyTorch>=1.8
    # (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
    @staticmethod
    @torch.jit.script
    def _lse_uv_inner(
        sim_1a,
        sim_a2,
        sign_a2,
        sim_corr,
        corr_sign,
        corr_batch_idx,
        corr_idx1,
        corr_idx2,
        corr_idx1_nooffset,
        corr_idx2_nooffset,
        norms1_batch_idx,
        norms1_idx,
        norms2_batch_idx,
        norms2_idx,
        norms,
        vec,
        dim: int,
    ):
        _, max_points, _ = sim_1a.shape

        if dim % 3 == 2:
            mat_inner1, sign_inner1 = logsumexp_signed_to_signed(
                sim_a2 + vec[:, None, :max_points], sign_a2, dim=2
            )
            mat_inner = sim_1a + mat_inner1[:, None, :]
            sign_inner = sign_inner1[:, None, :]
        else:
            mat_inner1 = torch.logsumexp(vec[:, :max_points, None] + sim_1a, dim=1)
            mat_inner = mat_inner1[:, :, None] + sim_a2
            sign_inner = sign_a2

        max_inner = mat_inner.max(dim).values
        max_outer = vec[:, max_points:] - norms
        lse_offset = torch.max(max_inner, max_outer)

        if dim % 3 == 2:
            correction = sim_corr + vec[corr_batch_idx, corr_idx2_nooffset]
            max_corr = segment_coo(
                correction, corr_idx1, dim_size=norms1_batch_idx.shape[0], reduce="max"
            )
            lse_offset[norms1_batch_idx, norms1_idx] = torch.max(
                max_corr, lse_offset[norms1_batch_idx, norms1_idx]
            )
        else:
            correction = sim_corr + vec[corr_batch_idx, corr_idx1_nooffset]
            max_corr = scatter(
                correction, corr_idx2, dim_size=norms2_batch_idx.shape[0], reduce="max"
            )
            lse_offset[norms2_batch_idx, norms2_idx] = torch.max(
                max_corr, lse_offset[norms2_batch_idx, norms2_idx]
            )

        sum_inner = (sign_inner * torch.exp(mat_inner - lse_offset.unsqueeze(dim))).sum(
            dim
        )
        sum_outer = torch.exp(-norms - lse_offset + vec[:, max_points:])

        if dim % 3 == 2:
            corr_offset = segment_coo(
                corr_sign
                * torch.exp(
                    correction - lse_offset[corr_batch_idx, corr_idx1_nooffset]
                ),
                corr_idx1,
                dim_size=norms1_batch_idx.shape[0],
                reduce="sum",
            )
            sum_inner[norms1_batch_idx, norms1_idx] += corr_offset
        else:
            corr_offset = scatter(
                corr_sign
                * torch.exp(
                    correction - lse_offset[corr_batch_idx, corr_idx2_nooffset]
                ),
                corr_idx2,
                dim_size=norms2_batch_idx.shape[0],
                reduce="sum",
            )
            sum_inner[norms2_batch_idx, norms2_idx] += corr_offset

        return torch.log(sum_inner + sum_outer) + lse_offset

    @staticmethod
    @torch.jit.script
    def _lse_uv_outer(norms: torch.Tensor, vec: torch.Tensor):
        _, max_points = norms.shape

        inner_part = -norms + vec[:, :max_points]
        logsum_outer = torch.logsumexp(vec[:, max_points:], dim=-1)

        lse_offset = torch.max(inner_part, logsum_outer[:, None])

        sum_inner = (inner_part - lse_offset).exp()
        sum_outer = (logsum_outer[:, None] - lse_offset).exp()

        return torch.log(sum_inner + sum_outer) + lse_offset

    @staticmethod
    def forward(
        ctx,
        cost_1a: torch.FloatTensor,
        sim_a2_scaled: torch.FloatTensor,
        sign_a2: torch.FloatTensor,
        cost_exact: torch.FloatTensor,
        sim_approx_scaled: torch.FloatTensor,
        sign_approx: torch.FloatTensor,
        corr_batch_idx: torch.LongTensor,
        corr_idx1: torch.LongTensor,
        corr_idx2: torch.LongTensor,
        corr_idx1_nooffset: torch.LongTensor,
        corr_idx2_nooffset: torch.LongTensor,
        norms1_batch_idx: torch.LongTensor,
        norms1_idx: torch.LongTensor,
        norms2_batch_idx: torch.LongTensor,
        norms2_idx: torch.LongTensor,
        norms1: torch.FloatTensor,
        norms2: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size, max_points = norms1.shape

        sim_1a_scaled = -cost_1a / sinkhorn_reg[:, None, None]
        norms1_scaled = norms1 / sinkhorn_reg[:, None]
        norms2_scaled = norms2 / sinkhorn_reg[:, None]

        sim_exact_scaled = -cost_exact / sinkhorn_reg[corr_batch_idx]
        sim_corr, corr_sign = logdiffexp(
            sim_exact_scaled + 1e-40, sim_approx_scaled + 1e-40, sign_approx
        )

        mask_n1 = (
            torch.arange(max_points, dtype=torch.int64, device=norms1.device).expand_as(
                norms1
            )
            >= num_points[0, :, None]
        )
        mask_n2 = (
            torch.arange(max_points, dtype=torch.int64, device=norms2.device).expand_as(
                norms2
            )
            >= num_points[1, :, None]
        )
        mask_u = torch.cat((mask_n1, mask_n2), dim=1)
        mask_v = torch.cat((mask_n2, mask_n1), dim=1)

        u = norms1.new_zeros(batch_size, 2 * max_points)
        v = norms1.new_zeros(batch_size, 2 * max_points)
        for _ in range(niter):
            u[:, :max_points] = -LogLCNSinkhornBP._lse_uv_inner(
                sim_1a_scaled,
                sim_a2_scaled,
                sign_a2,
                sim_corr,
                corr_sign,
                corr_batch_idx,
                corr_idx1,
                corr_idx2,
                corr_idx1_nooffset,
                corr_idx2_nooffset,
                norms1_batch_idx,
                norms1_idx,
                norms2_batch_idx,
                norms2_idx,
                norms1_scaled,
                v,
                dim=2,
            )
            u[:, max_points:] = -LogLCNSinkhornBP._lse_uv_outer(norms2_scaled, v)
            u.masked_fill_(mask_u, -1e10)
            v[:, :max_points] = -LogLCNSinkhornBP._lse_uv_inner(
                sim_1a_scaled,
                sim_a2_scaled,
                sign_a2,
                sim_corr,
                corr_sign,
                corr_batch_idx,
                corr_idx1,
                corr_idx2,
                corr_idx1_nooffset,
                corr_idx2_nooffset,
                norms1_batch_idx,
                norms1_idx,
                norms2_batch_idx,
                norms2_idx,
                norms2_scaled,
                u,
                dim=1,
            )
            v[:, max_points:] = -LogLCNSinkhornBP._lse_uv_outer(norms1_scaled, u)
            v.masked_fill_(mask_v, -1e10)

        T12 = torch.exp(-norms1_scaled + u[:, :max_points] + v[:, max_points:])
        T21 = torch.exp(-norms2_scaled + u[:, max_points:] + v[:, :max_points])
        T22_u = torch.exp(u[:, max_points:] + v[:, max_points:].logsumexp(-1)[:, None])
        T22_v = torch.exp(u[:, max_points:].logsumexp(-1)[:, None] + v[:, max_points:])

        T1a_log = sim_1a_scaled + u[:, :max_points, None]
        Ta1_log = sim_a2_scaled + v[:, None, :max_points]

        T1a_logsum = torch.logsumexp(T1a_log, dim=1)
        Ta1_logsum, Ta1_sum_sign = logsumexp_signed_to_signed(Ta1_log, sign_a2, dim=2)

        T11_sumright = (
            Ta1_sum_sign[:, None, :] * (T1a_log + Ta1_logsum[:, None, :]).exp()
        )
        T11_sumleft = sign_a2 * (T1a_logsum[:, :, None] + Ta1_log).exp()

        C11 = sinkhorn_reg * (
            (u[:, :max_points] * T11_sumright.sum(2)).sum(1)
            + (v[:, :max_points] * T11_sumleft.sum(1)).sum(1)
        )
        C12 = sinkhorn_reg * (
            (u[:, :max_points] * T12).sum(1) + (v[:, max_points:] * T12).sum(1)
        )
        C21 = sinkhorn_reg * (
            (u[:, max_points:] * T21).sum(1) + (v[:, :max_points] * T21).sum(1)
        )
        C22 = sinkhorn_reg * (
            (u[:, max_points:] * T22_u).sum(1) + (v[:, max_points:] * T22_v).sum(1)
        )
        C_nystrom = C11 + C12 + C21 + C22

        T11_exact = torch.exp(
            sim_exact_scaled
            + u[corr_batch_idx, corr_idx1_nooffset]
            + v[corr_batch_idx, corr_idx2_nooffset]
        )
        T11_approx = torch.exp(
            sim_approx_scaled
            + u[corr_batch_idx, corr_idx1_nooffset]
            + v[corr_batch_idx, corr_idx2_nooffset]
        )
        T11_delta = T11_exact - T11_approx
        C11_corr = sinkhorn_reg * (
            segment_coo(
                u[norms1_batch_idx, norms1_idx]
                * segment_coo(
                    T11_delta,
                    corr_idx1,
                    dim_size=norms1_batch_idx.shape[0],
                    reduce="sum",
                ),
                norms1_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
            + segment_coo(
                v[norms2_batch_idx, norms2_idx]
                * scatter(
                    T11_delta,
                    corr_idx2,
                    dim_size=norms2_batch_idx.shape[0],
                    reduce="sum",
                ),
                norms2_batch_idx,
                dim_size=batch_size,
                reduce="sum",
            )
        )
        C = C_nystrom + C11_corr

        ctx.save_for_backward(
            T11_sumleft,
            T11_sumright,
            T11_exact,
            T11_approx,
            corr_batch_idx,
            T12,
            T21,
            sinkhorn_reg,
        )

        return C

    @staticmethod
    def backward(ctx, grad_output):
        (
            T11_sumleft,
            T11_sumright,
            T11_exact,
            T11_approx,
            corr_batch_idx,
            T12,
            T21,
            sinkhorn_reg,
        ) = ctx.saved_tensors

        return (
            T11_sumright * grad_output[:, None, None],
            -sinkhorn_reg[:, None, None] * T11_sumleft * grad_output[:, None, None],
            None,
            T11_exact * grad_output[corr_batch_idx],
            sinkhorn_reg[corr_batch_idx] * T11_approx * grad_output[corr_batch_idx],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            T12 * grad_output[:, None],
            T21 * grad_output[:, None],
            None,
            None,
            None,
        )
