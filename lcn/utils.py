import math
from inspect import signature

import torch
import torch_scatter

__all__ = [
    "log_signed",
    "logsumexp_signed",
    "logsumexp_signed_to_signed",
    "logdiffexp",
    "loginvexp",
    "scatter",
    "segment_coo",
    "segment_csr",
    "repeat_blocks",
    "get_matching_embeddings",
    "get_function_args",
    "call_with_filtered_args",
]


@torch.jit.script
def log_signed(mat: torch.Tensor):
    mat_neg = mat < 0
    sign = (
        torch.ones_like(mat, device=mat.device, dtype=mat.dtype, layout=mat.layout)
        - 2 * mat_neg
    )
    mat_log = mat.abs().log()
    return mat_log, sign


@torch.jit.script
def logsumexp_signed(
    mat: torch.Tensor, sign: torch.Tensor, dim: int, check_sign: bool = True
):
    offset = mat.max(dim).values
    offset = torch.clamp(offset, min=-1e10, max=1e10)
    mat_e = (mat - offset.unsqueeze(dim)).exp()
    mat_sum = (mat_e * sign).sum(dim)
    mat_sum += 1e-40
    res = mat_sum.abs().log() + offset
    if check_sign:
        assert torch.all(
            res.exp()[mat_sum < 0] < 1e-7
        ), "Negative values in result. Use `logsumexp_signed_to_signed`."
    return res


# For some reason scripting this is broken in PyTorch>=1.8
# (silently exits in 1.8, "bad_variant_access" in 1.9, don't know why)
@torch.jit.script
def logsumexp_signed_to_signed(mat: torch.Tensor, sign: torch.Tensor, dim: int):
    offset = mat.max(dim).values
    offset = torch.clamp(offset, min=-1e10, max=1e10)
    mat_e = (mat - offset.unsqueeze(dim)).exp()
    mat_sum = (mat_e * sign).sum(dim)
    mat_sum += 1e-40
    mat_sum_abs, sign_res = log_signed(mat_sum)
    return mat_sum_abs + offset, sign_res


@torch.jit.script
def logdiffexp(tensor1: torch.Tensor, tensor2: torch.Tensor, sign2: torch.Tensor):
    lse_offset = torch.max(tensor1, tensor2)
    diff = torch.exp(tensor1 - lse_offset) - sign2 * torch.exp(tensor2 - lse_offset)
    res_without_offset, res_sign = log_signed(diff + 1e-40)
    return res_without_offset + lse_offset, res_sign


def loginvexp(mat, use_double=False):
    # Offsets seem to only make things worse.
    if use_double:
        mat = mat.double()
    exp_inv = torch.exp(mat).inverse()

    inv, inv_sign = log_signed(exp_inv)
    if use_double:
        inv = inv.float()
        inv_sign = inv_sign.float()

    # Negative entries below machine precision are likely errors.
    # log(2**(-53)) = -36.7, but let's be more lenient than that.
    inv_sign[inv < -100] = 1.0

    return inv, inv_sign


@torch.jit.script
def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    dim: int = -1,
    fill_value: float = math.nan,
    reduce: str = "sum",
):
    if torch.isnan(torch.tensor(fill_value)):
        if reduce == "max":
            fill_value = -1e38  # torch.finfo(src.dtype).min
        elif reduce == "min":
            fill_value = 1e38  # torch.finfo(src.dtype).max

    if torch.isnan(torch.tensor(fill_value)):
        if reduce == "logsumexp":
            return torch_scatter.composite.scatter_logsumexp(
                src, index, dim=dim, dim_size=dim_size
            )
        else:
            return torch_scatter.scatter(
                src, index, dim=dim, dim_size=dim_size, reduce=reduce
            )
    else:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
        if reduce == "logsumexp":
            torch_scatter.composite.scatter_logsumexp(src, index, dim=dim, out=out)
        else:
            torch_scatter.scatter(src, index, dim=dim, out=out, reduce=reduce)
        return out


@torch.jit.script
def segment_coo(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    fill_value: float = math.nan,
    reduce: str = "sum",
):
    if torch.isnan(torch.tensor(fill_value)):
        if reduce == "max":
            fill_value = -1e38  # torch.finfo(src.dtype).min
        elif reduce == "min":
            fill_value = 1e38  # torch.finfo(src.dtype).max

    if torch.isnan(torch.tensor(fill_value)):
        return torch_scatter.segment_coo(src, index, dim_size=dim_size, reduce=reduce)
    else:
        shape = src.shape[:-1] + (dim_size,)
        out = torch.full(shape, fill_value, dtype=src.dtype, device=src.device)
        torch_scatter.segment_coo(src, index, out=out, reduce=reduce)
        return out


@torch.jit.script
def segment_csr(
    src: torch.Tensor,
    indptr: torch.Tensor,
    fill_value: float = math.nan,
    reduce: str = "sum",
):
    if torch.isnan(torch.tensor(fill_value)):
        if reduce == "max":
            fill_value = -1e38  # torch.finfo(src.dtype).min
        elif reduce == "min":
            fill_value = 1e38  # torch.finfo(src.dtype).max

    if torch.isnan(torch.tensor(fill_value)):
        return torch_scatter.segment_csr(src, indptr, reduce=reduce)
    else:
        out = torch_scatter.segment_csr(src, indptr, reduce=reduce)
        mask = indptr[1:] == indptr[:-1]
        out.masked_fill_(mask, fill_value)
        return out


def repeat_blocks(sizes, repeats, continuous_indexing=True):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
    else:
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if continuous_indexing:
        if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
            # If a group was skipped (repeats=0) we need to add its size
            diffs = r1[1:] - r1[:-1]
            indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
            insert_val += torch_scatter.segment_csr(
                sizes[: r1[-1]], indptr, reduce="sum"
            )
        else:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[r1[1:] != r1[:-1]] = 1

    # Assign index-offseting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def get_matching_embeddings(X, dist):
    dists = dist.cdist(X, X)
    matches = torch.isclose(dists, X.new_zeros(1), atol=1e-5)
    batch, X1, X2 = torch.where(matches)
    match_nonself = torch.where(X1 < X2)
    return batch[match_nonself], X1[match_nonself], X2[match_nonself]


def get_function_args(function, **kwargs):
    if isinstance(function, torch.autograd.function.FunctionMeta):
        return [
            kwargs[key]
            for key in signature(function.forward).parameters.keys()
            if key != "ctx"
        ]
    elif isinstance(function, torch.jit.ScriptFunction):
        return [kwargs[arg.name] for arg in function.schema.arguments]
    else:
        return [kwargs[key] for key in signature(function).parameters.keys()]


def call_with_filtered_args(function, **kwargs):
    args = get_function_args(function, **kwargs)
    if isinstance(function, torch.autograd.function.FunctionMeta):
        return function.apply(*args)
    else:
        return function(*args)
