import torch


# Scripting only works with Python>=1.5 (don't know why)
# @torch.jit.script
# @torch.no_grad()
def get_angular_buckets(
    X: torch.Tensor, num_points: torch.Tensor, num_buckets: int, num_hashes: int
):
    """
    Paper: https://arxiv.org/abs/1509.02897
    Adapted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_reformer.py
    """
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2

    # We sample a different random rotation for each round of hashing to
    # decrease the probability of hash misses.
    assert (
        num_buckets % 2 == 0
    ), f"There should be an even number of buckets, but `num_buckets`: {num_buckets}"
    rotation_size = num_buckets

    # create a random emb_dim x num_hashes x num_buckets/2 tensor
    rotations_shape = (X.shape[-1], num_hashes, rotation_size // 2)
    random_rotations = torch.randn(rotations_shape, device=X.device, dtype=X.dtype)

    # Output dim: Batch_Size x num_hashes x Seq_Len x num_buckets/2
    rotated_vectors = torch.einsum("btd,dhr->bhtr", X, random_rotations)

    rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
    buckets = torch.argmax(rotated_vectors, dim=-1)

    # Mask padding entries
    mask_padding = torch.zeros(
        (batch_size, num_hashes, 2 * max_points), dtype=torch.bool, device=X.device
    )
    mask_padding[:, :, :max_points] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, num_hashes, -1
        )
        >= num_points[0, :, None, None]
    )
    mask_padding[:, :, max_points:] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, num_hashes, -1
        )
        >= num_points[1, :, None, None]
    )
    buckets.masked_fill_(mask_padding, num_buckets)

    return buckets


def get_line_buckets(
    X: torch.Tensor, num_points: torch.Tensor, num_buckets: int, num_hashes: int
):
    batch_size, max_points2, emb_size = X.shape
    max_points = max_points2 // 2

    # Sample random directions in the embedding space. We don't normalize the directions
    # because we normalize further down anyway when we split the projections into buckets.
    lines = torch.randn((num_hashes, emb_size), device=X.device, dtype=X.dtype)

    # Project the data onto the lines
    projections = torch.einsum("btd,hd->bht", X, lines)

    # Normalize the line segments containing the projections per batch
    eps = X.new_full((), 10 ** -10)
    normalized = projections - projections.min(dim=-1, keepdim=True).values
    normalized = normalized / (normalized.max(dim=-1, keepdim=True).values).max(eps)

    # Divide the line segments into equal width buckets
    buckets = (normalized * num_buckets).floor().long()

    # Ensure that we get buckets between 0 and (num_buckets - 1)
    buckets = torch.where(
        buckets == num_buckets,
        X.new_full((), num_buckets - 1, dtype=torch.long),
        buckets,
    )

    # Mask padding entries
    mask_padding = X.new_empty(
        (batch_size, num_hashes, 2 * max_points), dtype=torch.bool
    )
    mask_padding[:, :, :max_points] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, num_hashes, -1
        )
        >= num_points[0, :, None, None]
    )
    mask_padding[:, :, max_points:] = (
        torch.arange(max_points, dtype=torch.long, device=X.device).expand(
            batch_size, num_hashes, -1
        )
        >= num_points[1, :, None, None]
    )
    buckets.masked_fill_(mask_padding, num_buckets)

    return buckets
