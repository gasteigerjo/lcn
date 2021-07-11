import math

import torch


class Sinkhorn(torch.autograd.Function):
    """Wasserstein distance with entropy regularization.
    Irregular dimensions due to batching are padded according to num_points.

    See Cuturi 2013, Sinkhorn Distances - Lightspeed Computation of Optimal Transport
    and Alg. 1 in Frogner 2015, Learning with a Wasserstein Loss.
    In our case h(x) = y = 1.

    Also inspired by Mocha.jl (https://github.com/pluskid/Mocha.jl/blob/master/src/layers/wasserstein-loss.jl)
    and the Python Optimal Transport library (https://github.com/rflamary/POT/blob/master/ot/bregman.py).

    sinkhorn_reg = 1 / lambda presents a trade-off: Smaller values are closer to the true EMD but converge more slowly and are less stable
    """

    @staticmethod
    def forward(
        ctx,
        cost_mat: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
    ):
        batch_size, max_points, _ = cost_mat.shape
        K = torch.exp(-cost_mat / sinkhorn_reg[:, None, None])
        if K.min() < 1e-13:
            raise RuntimeError(f"Sinkhorn reg too small. Smallest K = {K.min()}")

        # Fill padding with 0
        mask_outer1 = (
            torch.arange(max_points, dtype=torch.float32, device=cost_mat.device)[
                :, None
            ].expand_as(cost_mat)
            >= num_points[:, None, None]
        )
        mask_outer2 = (
            torch.arange(
                max_points, dtype=torch.float32, device=cost_mat.device
            ).expand_as(cost_mat)
            >= num_points[:, None, None]
        )
        mask_outer = mask_outer1 | mask_outer2
        K = K.masked_fill(mask_outer, 0)

        # Mask that acts only on K's first column
        mask_vector = (
            torch.arange(
                max_points, dtype=torch.float32, device=cost_mat.device
            ).expand(batch_size, max_points)
            >= num_points[:, None]
        ).float()
        K_u = K.clone().transpose(1, 2)
        K_u[:, :, 0] += mask_vector

        K_v = K.clone()
        K_v[:, :, 0] += mask_vector

        u = cost_mat.new_full([batch_size, max_points], 1 / max_points)
        for _ in range(niter):
            v = 1 / torch.einsum("bij, bj -> bi", K_u, u)
            u = 1 / torch.einsum("bij, bj -> bi", K_v, v)

        if (
            torch.isnan(u).any()
            or torch.isnan(v).any()
            or u.max() > 1e10
            or v.max() > 1e10
        ):
            raise RuntimeError(
                f"Excessively large/nan values: u.max = {u.max():.2e}, v.max = {v.max():.2e}"
            )

        T = torch.diag_embed(u) @ K @ torch.diag_embed(v)

        ctx.save_for_backward(T)

        return torch.sum(cost_mat * T, dim=[1, 2])

    @staticmethod
    def backward(ctx, grad_output):
        (T,) = ctx.saved_tensors
        return T * grad_output[:, None, None], None, None, None, None


class LogSinkhorn(torch.autograd.Function):
    """Wasserstein distance with entropy regularization, calculated in log space.
    Irregular dimensions due to batching are padded according to num_points.

    sinkhorn_reg = 1 / lambda presents a trade-off: Smaller values are closer to the true EMD but converge more slowly and are less stable
    """

    @staticmethod
    def forward(
        ctx,
        cost_mat: torch.FloatTensor,
        num_points: torch.LongTensor,
        sinkhorn_reg: torch.FloatTensor,
        niter: int = 50,
        offset_entropy: bool = True,
    ):

        T_log, u, v = arg_log_sinkhorn(cost_mat, num_points, sinkhorn_reg, niter)
        T = T_log.exp()

        ctx.save_for_backward(T)

        if offset_entropy:
            return sinkhorn_reg * (
                (torch.clamp(u, max=1e10) * T.sum(2)).sum(1)
                + (torch.clamp(v, max=1e10) * T.sum(1)).sum(1)
            )
        else:
            # Original variant, as in Cuturi 2013: C * T
            return torch.sum(torch.clamp(cost_mat, max=1e10) * T, dim=[1, 2])

    @staticmethod
    def backward(ctx, grad_output):
        (T,) = ctx.saved_tensors
        return T * grad_output[:, None, None], None, None, None, None


@torch.jit.script
def arg_log_sinkhorn(
    cost_mat: torch.Tensor,
    num_points: torch.Tensor,
    sinkhorn_reg: torch.Tensor,
    niter: int = 50,
):
    """Transport matrix according to Wasserstein distance with entropy regularization, calculated in log space.
    Irregular dimensions due to batching are padded according to num_points.

    See Peyré, SinkhornAutoDiff (https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py)
    and Daza, Approximating Wasserstein distances with PyTorch (https://dfdazac.github.io/sinkhorn.html)
    In our case mu = nu = 1.
    """
    batch_size, max_points, _ = cost_mat.shape

    # Fill padding with inf
    if num_points.dim() == 1:
        # Quadratic cost matrix
        mask_outer1 = (
            torch.arange(max_points, dtype=torch.int64, device=cost_mat.device)[
                :, None
            ].expand_as(cost_mat)
            >= num_points[:, None, None]
        )
        mask_outer2 = (
            torch.arange(
                max_points, dtype=torch.int64, device=cost_mat.device
            ).expand_as(cost_mat)
            >= num_points[:, None, None]
        )
    else:
        # Non-quadratic cost matrix
        mask_outer1 = (
            torch.arange(max_points, dtype=torch.int64, device=cost_mat.device)[
                :, None
            ].expand_as(cost_mat)
            >= num_points[0, :, None, None]
        )
        mask_outer2 = (
            torch.arange(
                max_points, dtype=torch.int64, device=cost_mat.device
            ).expand_as(cost_mat)
            >= num_points[1, :, None, None]
        )
    mask_outer = mask_outer1 | mask_outer2
    cost_inf = cost_mat.masked_fill(mask_outer, math.inf) / sinkhorn_reg[:, None, None]

    u = cost_mat.new_zeros(batch_size, max_points)
    v = cost_mat.new_zeros(batch_size, max_points)
    for _ in range(niter):
        # clamp to prevent NaN for inf - inf
        u = -torch.logsumexp(torch.clamp(v[:, None, :], max=1e10) - cost_inf, dim=-1)
        v = -torch.logsumexp(torch.clamp(u[:, :, None], max=1e10) - cost_inf, dim=-2)

    T_log = torch.clamp(u[:, :, None] + v[:, None, :], max=1e10) - cost_inf

    return T_log, u, v


@torch.jit.script
def arg_log_sinkhorn2(
    cost_mat: torch.Tensor,
    num_points: torch.Tensor,
    sinkhorn_reg: torch.Tensor,
    niter: int = 50,
):
    """Transport matrix according to Wasserstein distance with entropy regularization, calculated in log space.
    Irregular dimensions due to batching are padded according to num_points.

    More stable than arg_log_sinkhorn due to repeated multiplication and division by sinkhorn_reg.
    """
    batch_size, max_points, _ = cost_mat.shape

    # Fill padding with inf
    if num_points.dim() == 1:
        # Quadratic cost matrix
        mask_outer1 = (
            torch.arange(max_points, dtype=torch.int64, device=cost_mat.device)[
                :, None
            ].expand_as(cost_mat)
            >= num_points[:, None, None]
        )
        mask_outer2 = (
            torch.arange(
                max_points, dtype=torch.int64, device=cost_mat.device
            ).expand_as(cost_mat)
            >= num_points[:, None, None]
        )
    else:
        # Non-quadratic cost matrix
        mask_outer1 = (
            torch.arange(max_points, dtype=torch.int64, device=cost_mat.device)[
                :, None
            ].expand_as(cost_mat)
            >= num_points[0, :, None, None]
        )
        mask_outer2 = (
            torch.arange(
                max_points, dtype=torch.int64, device=cost_mat.device
            ).expand_as(cost_mat)
            >= num_points[1, :, None, None]
        )
    mask_outer = mask_outer1 | mask_outer2
    cost_inf = cost_mat.masked_fill(mask_outer, math.inf)

    u = cost_mat.new_zeros(batch_size, max_points)
    v = cost_mat.new_zeros(batch_size, max_points)
    for _ in range(niter):
        u = -sinkhorn_reg[:, None] * torch.logsumexp(
            torch.clamp(v[:, None, :] - cost_inf, min=-1e10)
            / sinkhorn_reg[:, None, None],
            dim=-1,
        )
        v = -sinkhorn_reg[:, None] * torch.logsumexp(
            torch.clamp(u[:, :, None] - cost_inf, min=-1e10)
            / sinkhorn_reg[:, None, None],
            dim=-2,
        )

    T_log = (
        torch.clamp(u[:, :, None] + v[:, None, :] - cost_inf, min=-1e10)
        / sinkhorn_reg[:, None, None]
    )
    return T_log, u, v
