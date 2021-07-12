# Partially based on https://github.com/facebookresearch/fastText/blob/master/alignment/unsup_align.py

import logging

from tqdm.autonotebook import tqdm
import torch

from lcn.alignment_utils import (
    procrustes,
    sqrt_eig,
)
from lcn.cost import distances, kernels
from lcn.cost.cost_matrix import get_cost_matrix
from lcn.sinkhorn import arg_log_sinkhorn2
from lcn.transport_plan import get_transport_plan


def convex_init(X, Y, sinkhorn_reg, niter=100, apply_sqrt=False, disable_tqdm=False):
    n, _ = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = X @ X.T, Y @ Y.T
    K_Y *= K_X.norm() / K_Y.norm()
    K2_X, K2_Y = K_X @ K_X, K_Y @ K_Y
    P = X.new_full([n, n], fill_value=1 / n)
    num_points = torch.tensor(P.shape[-1:], device=X.device)
    for it in tqdm(range(niter), disable=disable_tqdm):
        G = P @ K2_X + K2_Y @ P - 2 * K_Y @ (P @ K_X)
        q_log, _, _ = arg_log_sinkhorn2(
            G.unsqueeze(0), num_points, sinkhorn_reg.unsqueeze(0), niter=5
        )
        q = q_log.exp().squeeze(0)
        alpha = 2.0 / (3.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = torch.norm(P @ K_X - K_Y @ P)
    logging.info(f"Objective: {obj.item():.3f}")
    return procrustes(P @ X, Y).T


def objective(X, Y, R, sinkhorn_reg, n=5000):
    Xn, Yn = X[:n], Y[:n]
    C = -(Xn @ R) @ Yn.T
    num_points = torch.tensor(C.shape[-1:], device=R.device)
    P_log, _, _ = arg_log_sinkhorn2(
        C.unsqueeze(0), num_points, sinkhorn_reg.unsqueeze(0), niter=10
    )
    P = P_log.exp().squeeze(0)
    return 1000 * torch.norm(Xn @ R - P @ Yn) / n


def align(
    X,
    Y,
    R,
    sinkhorn_reg,
    method,
    nystrom,
    sparse,
    lr=10.0,
    niter=200,
    lr_half_niter=50,
    ntrain=20000,
    print_niter=50,
    disable_tqdm=False,
):
    xt, yt = X[:ntrain], Y[:ntrain]
    num_embs_rep = torch.tensor(ntrain, device=R.device).repeat(2, 1)
    tq = tqdm(range(niter), disable=disable_tqdm)
    for it in tq:
        # compute OT on minibatch
        embeddings = [xt.unsqueeze(0), (yt @ R.T).unsqueeze(0)]
        cost_matrix = get_cost_matrix(
            embeddings,
            num_points=num_embs_rep,
            nystrom=nystrom,
            sparse=sparse,
            sinkhorn_reg=sinkhorn_reg.unsqueeze(0),
            sinkhorn_niter=10,
            alpha=X.new_ones(1),
            dist=distances.KernelDist(kernels.Dot()),
            dist_cluster=distances.Cosine(),
            bp_cost_matrix=False,
        )
        mv_prod = get_transport_plan(cost_matrix, niter=10, method=method)
        # compute gradient
        del cost_matrix
        G = -xt.T @ mv_prod(yt)
        del mv_prod
        R -= lr / ntrain * G
        # project on orthogonal matrices
        U, _, V = torch.svd(R, some=False)
        R = U @ V.T
        if it > 0 and it % lr_half_niter == 0:
            lr /= 2
        if it > 0 and it % print_niter == 0:
            tq.set_postfix(obj=f"{objective(xt, yt, R, sinkhorn_reg):.3f} (iter {it})")
    return R


def align_original(
    X,
    Y,
    R,
    sinkhorn_reg,
    lr=500.0,
    bsz=500,
    nepoch=5,
    niter=5000,
    nmax=20000,
    disable_tqdm=False,
):
    for epoch in tqdm(range(1, nepoch + 1), disable=disable_tqdm):
        num_points = torch.tensor([bsz], device=R.device)
        for _ in tqdm(range(1, niter + 1), disable=disable_tqdm):
            # sample mini-batch
            xt = X[torch.randperm(nmax, device=R.device)[:bsz], :]
            yt = Y[torch.randperm(nmax, device=R.device)[:bsz], :]
            # compute OT on minibatch
            C = -(xt @ R) @ yt.T
            P_log, _, _ = arg_log_sinkhorn2(
                C.unsqueeze(0), num_points, sinkhorn_reg.unsqueeze(0), niter=10
            )
            P = P_log.exp().squeeze(0)
            # compute gradient
            G = -xt.T @ (P @ yt)
            R -= lr / bsz * G
            # project on orthogonal matrices
            U, _, V = torch.svd(R, some=False)
            R = U @ V.T
        bsz *= 2
        niter //= 4
        logging.debug(f"Epoch: {epoch}  obj: {objective(X, Y, R, sinkhorn_reg):.3f}")
    return R
