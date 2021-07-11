import math

import torch


@torch.jit.script
def entropyRegLogSinkhorn(
    cost_mat: torch.Tensor,
    num_points: torch.Tensor,
    sinkhorn_reg: torch.Tensor,
    reg_marginal: torch.Tensor,
    niter: int = 50,
    offset_entropy: bool = True,
):
    """
    Unbalanced Wasserstein distance with entropy regularization, calculated in log space.
    See Chizat 2018, Scaling Algorithms for Unbalanced Optimal Transport Problems, Alg. 2
    (but all in log space, dx=1/n, dy=1/m, p=1/n, q=1/m)

    Irregular dimensions due to batching are padded according to num_points.
    """
    batch_size, max_points, _ = cost_mat.shape

    # Fill padding with inf
    mask_outer1 = (
        torch.arange(max_points, dtype=torch.int64, device=cost_mat.device)[
            :, None
        ].expand_as(cost_mat)
        >= num_points[0, :, None, None]
    )
    mask_outer2 = (
        torch.arange(max_points, dtype=torch.int64, device=cost_mat.device).expand_as(
            cost_mat
        )
        >= num_points[1, :, None, None]
    )
    mask_outer = mask_outer1 | mask_outer2
    cost_inf = cost_mat.masked_fill(mask_outer, math.inf) / sinkhorn_reg[:, None, None]

    prefactor = -reg_marginal[:, None] / (reg_marginal[:, None] + sinkhorn_reg[:, None])
    summand = (
        torch.log(num_points[0, :].float()) - torch.log(num_points[1, :].float())
    )[:, None]

    u = cost_mat.new_zeros(batch_size, max_points)
    v = cost_mat.new_zeros(batch_size, max_points)
    for _ in range(niter):
        u = prefactor * (
            summand
            + torch.logsumexp(torch.clamp(v[:, None, :] - cost_inf, min=-1e10), dim=-1)
        )
        v = prefactor * (
            -summand
            + torch.logsumexp(torch.clamp(u[:, :, None] - cost_inf, min=-1e10), dim=-2)
        )

    T = torch.exp(torch.clamp(u[:, :, None] + v[:, None, :] - cost_inf, min=-1e10))

    if offset_entropy:
        return sinkhorn_reg * (
            (torch.clamp(u, max=1e10) * T.sum(2)).sum(1)
            + (torch.clamp(v, max=1e10) * T.sum(1)).sum(1)
        )
    else:
        # Original variant, as in Cuturi 2013: C * T
        return torch.sum(cost_mat * T, dim=[1, 2])


class LogSinkhornBP(torch.autograd.Function):
    """
    Wasserstein distance with entropy regularization
    using the BP matrix for unbalanced distributions.
    A.k.a. learnable unbalanced OT. Calculated in log space.
    """

    @staticmethod
    @torch.jit.script
    def _lse_uv_inner(cost, norms, vec, dim: int):
        _, max_points, _ = cost.shape

        if dim % 3 == 2:
            mat_inner = vec[:, None, :max_points] - cost
        else:
            mat_inner = vec[:, :max_points, None] - cost

        max_inner = mat_inner.max(dim).values
        max_outer = vec[:, max_points:] - norms
        lse_offset = torch.max(max_inner, max_outer)

        sum_inner = torch.exp(mat_inner - lse_offset.unsqueeze(dim)).sum(dim)
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
        cost_mat: torch.FloatTensor,
        norms1: torch.FloatTensor,
        norms2: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size, max_points = norms1.shape

        cost_scaled = cost_mat / sinkhorn_reg[:, None, None]
        norms1_scaled = norms1 / sinkhorn_reg[:, None]
        norms2_scaled = norms2 / sinkhorn_reg[:, None]

        cost_clamped = torch.clamp(cost_scaled, max=1e20)
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
            u[:, :max_points] = -LogSinkhornBP._lse_uv_inner(
                cost_clamped, norms1_clamped, v, dim=2
            )
            u[:, max_points:] = -LogSinkhornBP._lse_uv_outer(norms2_clamped, v)
            u.masked_fill_(mask_u, -1e10)
            v[:, :max_points] = -LogSinkhornBP._lse_uv_inner(
                cost_clamped, norms2_clamped, u, dim=1
            )
            v[:, max_points:] = -LogSinkhornBP._lse_uv_outer(norms1_clamped, u)
            v.masked_fill_(mask_v, -1e10)

        T11 = torch.exp(u[:, :max_points, None] + v[:, None, :max_points] - cost_scaled)
        T12 = torch.exp(-norms1_scaled + u[:, :max_points] + v[:, max_points:])
        T21 = torch.exp(-norms2_scaled + u[:, max_points:] + v[:, :max_points])
        T22_u = torch.exp(u[:, max_points:] + v[:, max_points:].logsumexp(-1)[:, None])
        T22_v = torch.exp(u[:, max_points:].logsumexp(-1)[:, None] + v[:, max_points:])

        C11 = sinkhorn_reg * (
            (u[:, :max_points] * T11.sum(2)).sum(1)
            + (v[:, :max_points] * T11.sum(1)).sum(1)
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

        ctx.save_for_backward(T11, T12, T21)

        return C

    @staticmethod
    def backward(ctx, grad_output):
        T11, T12, T21 = ctx.saved_tensors

        return (
            T11 * grad_output[:, None, None],
            T12 * grad_output[:, None],
            T21 * grad_output[:, None],
            None,
            None,
            None,
        )
