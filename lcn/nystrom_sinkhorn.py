import torch

from lcn.utils import logsumexp_signed, logsumexp_signed_to_signed


# For some reason scripting this is broken in PyTorch>=1.8
# (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
@torch.jit.script
def nystrom_lse_uv(sim_1a, sim_a2, sign_a2, vec, dim: int):
    """
    Just a helper function implementing the calculations inside the Sinkhorn loop.
    """
    if dim % 3 == 2:
        mat_inner1, sign_inner1 = logsumexp_signed_to_signed(
            sim_a2 + vec[:, None, :], sign_a2, dim=2
        )
        sum_inner = logsumexp_signed(
            sim_1a + mat_inner1[:, None, :], sign_inner1[:, None, :], dim=2
        )  # , check_sign=False)
    else:
        mat_inner1 = torch.logsumexp(vec[:, :, None] + sim_1a, dim=1)
        sum_inner = logsumexp_signed(
            mat_inner1[:, :, None] + sim_a2, sign_a2, dim=1
        )  # , check_sign=False)
    return sum_inner


def arg_log_nystrom_sinkhorn(
    cost_1a: torch.FloatTensor,
    sim_a2_scaled: torch.FloatTensor,
    sign_a2: torch.FloatTensor,
    num_points: torch.LongTensor,
    sinkhorn_reg: torch.FloatTensor,
    niter: int = 50,
):
    """
    Wasserstein distance with entropy regularization
    using a low-rank Nyström approximation of the cost matrix.
    Irregular dimensions due to batching are padded according to num_points.
    Calculated in log space.

    Arguments
    ---------
    cost_1a:            Padded left part of the Nyström approximation C = U @ V,
                        as a cost in log-space, i.e. U = exp(-cost_1a / sinkhorn_reg)
    sim_a2_scaled:      Padded right part of the Nyström approximation,
                        as a similarity (-cost) in log-space,
                        already divided by the regularization ("scaled")
    sign_a2:            Padded sign of the right part of the Nyström approximation,
                        i.e. V = sign_a2 * exp(sim_a2_scaled)
    num_points:         Number of points per side and sample, shape [2, batch_size]
    sinkhorn_reg:       Sinkhorn regularization
    niter:              Number of Sinkhorn iterations

    Returns
    -------
    T1a_log:            Padded left matrix of the decomposed transport plan
                        T = T1a @ Ta2 in log-space
    Ta2_log:            Padded right matrix of the decomposed transport plan
                        T = T1a @ Ta2 in log-space
    u:                  Resulting left normalization
    v:                  Resulting right normalization
    """
    batch_size, max_points = cost_1a.shape[:2]

    sim_1a_scaled = -cost_1a / sinkhorn_reg[:, None, None]

    u = num_points.new_zeros(batch_size, max_points)
    v = num_points.new_zeros(batch_size, max_points)
    for i in range(niter):
        u = -nystrom_lse_uv(sim_1a_scaled, sim_a2_scaled, sign_a2, v, dim=2)
        v = -nystrom_lse_uv(sim_1a_scaled, sim_a2_scaled, sign_a2, u, dim=1)

    T1a_log = sim_1a_scaled + u[:, :, None]
    Ta2_log = sim_a2_scaled + v[:, None, :]

    return T1a_log, Ta2_log, u, v


class LogNystromSinkhorn(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using a low-rank Nyström approximation of the cost matrix.
    Irregular dimensions due to batching are padded according to num_points.
    Calculated in log space.
    Call via LogNystromSinkhorn.apply(*args), without ctx argument.

    Arguments
    ---------
    cost_1a:            Padded left part of the Nyström approximation C = U @ V,
                        as a cost in log-space, i.e. U = exp(-cost_1a / sinkhorn_reg)
    sim_a2_scaled:      Padded right part of the Nyström approximation,
                        as a similarity (-cost) in log-space,
                        already divided by the regularization ("scaled")
    sign_a2:            Padded sign of the right part of the Nyström approximation,
                        i.e. V = sign_a2 * exp(sim_a2_scaled)
    num_points:         Number of points per side and sample, shape [2, batch_size]
    sinkhorn_reg:       Sinkhorn regularization
    niter:              Number of Sinkhorn iterations

    Returns
    -------
    C:                  Transport cost per sample
    """

    @staticmethod
    def forward(
        ctx,
        cost_1a: torch.FloatTensor,
        sim_a2_scaled: torch.FloatTensor,
        sign_a2: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):

        T1a_log, Ta2_log, u, v = arg_log_nystrom_sinkhorn(
            cost_1a, sim_a2_scaled, sign_a2, num_points, sinkhorn_reg, niter
        )

        T1a_logsum = torch.logsumexp(T1a_log, dim=1)
        Ta2_logsum, Ta2_sum_sign = logsumexp_signed_to_signed(Ta2_log, sign_a2, dim=2)

        T_sumright = Ta2_sum_sign[:, None, :] * (T1a_log + Ta2_logsum[:, None, :]).exp()
        T_sumleft = sign_a2 * (T1a_logsum[:, :, None] + Ta2_log).exp()

        C = sinkhorn_reg * (
            (u * T_sumright.sum(2)).sum(1) + (v * T_sumleft.sum(1)).sum(1)
        )

        ctx.save_for_backward(T_sumleft, T_sumright, sinkhorn_reg)

        return C

    @staticmethod
    def backward(ctx, grad_output):
        T_sumleft, T_sumright, sinkhorn_reg = ctx.saved_tensors

        return (
            T_sumright * grad_output[:, None, None],
            -sinkhorn_reg[:, None, None] * T_sumleft * grad_output[:, None, None],
            None,
            None,
            None,
            None,
        )


class LogNystromSinkhornBP(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using the BP matrix for unbalanced distributions
    and a low-rank Nyström approximation of the cost matrix.
    Irregular dimensions due to batching are padded according to num_points.
    Calculated in log space.
    Call via LogNystromSinkhornBP.apply(*args), without ctx argument.

    Arguments
    ---------
    cost_1a:            Padded left part of the Nyström approximation C = U @ V,
                        as a cost in log-space, i.e. U = exp(-cost_1a / sinkhorn_reg)
    sim_a2_scaled:      Padded right part of the Nyström approximation,
                        as a similarity (-cost) in log-space,
                        already divided by the regularization ("scaled")
    sign_a2:            Padded sign of the right part of the Nyström approximation,
                        i.e. V = sign_a2 * exp(sim_a2_scaled)
    norms1:             Padded norms of the left embeddings
    norms2:             Padded norms of the right embeddings
    num_points:         Number of points per side and sample, shape [2, batch_size]
    sinkhorn_reg:       Sinkhorn regularization
    niter:              Number of Sinkhorn iterations

    Returns
    -------
    C:                  Transport cost per sample
    """

    # For some reason scripting this is broken in PyTorch>=1.8
    # (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
    @staticmethod
    @torch.jit.script
    def _lse_uv_inner(sim_1a, sim_a2, sign_a2, norms, vec, dim: int):
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

        sum_inner = (sign_inner * torch.exp(mat_inner - lse_offset.unsqueeze(dim))).sum(
            dim
        )
        sum_outer = torch.exp(-norms - lse_offset + vec[:, max_points:])

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

        sim_1a_clamped = torch.clamp(sim_1a_scaled, min=-1e20)
        norms1_clamped = torch.clamp(norms1_scaled, max=1e20)
        norms2_clamped = torch.clamp(norms2_scaled, max=1e20)

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
        for i in range(niter):
            u[:, :max_points] = -LogNystromSinkhornBP._lse_uv_inner(
                sim_1a_clamped, sim_a2_scaled, sign_a2, norms1_clamped, v, dim=2
            )
            u[:, max_points:] = -LogNystromSinkhornBP._lse_uv_outer(norms2_clamped, v)
            u.masked_fill_(mask_u, -1e10)
            v[:, :max_points] = -LogNystromSinkhornBP._lse_uv_inner(
                sim_1a_clamped, sim_a2_scaled, sign_a2, norms2_clamped, u, dim=1
            )
            v[:, max_points:] = -LogNystromSinkhornBP._lse_uv_outer(norms1_clamped, u)
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
        C = C11 + C12 + C21 + C22

        ctx.save_for_backward(T11_sumleft, T11_sumright, T12, T21, sinkhorn_reg)

        return C

    @staticmethod
    def backward(ctx, grad_output):
        T11_sumleft, T11_sumright, T12, T21, sinkhorn_reg = ctx.saved_tensors

        return (
            T11_sumright * grad_output[:, None, None],
            -sinkhorn_reg[:, None, None] * T11_sumleft * grad_output[:, None, None],
            None,
            T12 * grad_output[:, None],
            T21 * grad_output[:, None],
            None,
            None,
            None,
        )
