import torch

from lcn.cost import distances
from lcn.cost.angular_lsh import get_angular_buckets
from lcn.cost.kmeans import (
    get_cluster_assignments,
    kmeans,
    kmeans_cat,
    kmeans_hier_inner,
)
from lcn.cost.sampling import kmeanspp_sampling_cat, uniform_sampling_cat


@torch.no_grad()
def get_landmarks(
    method,
    embeddings,
    num_points,
    num_clusters,
    dist,
    num_hashbands=1,
    num_hashes_per_band=1,
    return_assignments=False,
    separate=False,
):
    batch_size, max_points, emb_size = embeddings[0].shape
    num_hashes = num_hashbands * num_hashes_per_band

    if separate:
        assert method in ["kmeans", "kmeans_hier"]
        embeddings_cat = torch.cat(embeddings, dim=0)
        num_points = num_points.flatten()
    else:
        embeddings_cat = torch.cat(embeddings, dim=1)

    if method == "sampling_uniform":
        landmarkss = []
        for _ in range(num_hashes):
            landmarkss.append(
                uniform_sampling_cat(
                    embeddings_cat, num_points, num_clusters, dist=distances.PNorm(p=2)
                )
            )
        landmarks = torch.stack(landmarkss, dim=1)
        del landmarkss
        cluster_cat = None
    elif method == "sampling_kmeanspp":
        landmarkss = []
        for _ in range(num_hashes):
            landmarkss.append(
                kmeanspp_sampling_cat(
                    embeddings_cat, num_clusters, dist=dist, num_points=num_points
                )
            )
        landmarks = torch.stack(landmarkss, dim=1)
        del landmarkss
        cluster_cat = None
    elif method == "kmeans_hier":
        assert num_hashes == 1
        assert isinstance(num_clusters, list)
        num_clusters_above = torch.tensor(num_clusters[:-1]).cumprod(0)

        if separate:
            _, cluster_cat = kmeans(
                embeddings_cat,
                num_points,
                kclusters=num_clusters[0],
                niter=10,
                dist=dist,
                kmeanspp_init=True,
            )
        else:
            _, cluster_cat = kmeans_cat(
                embeddings_cat,
                num_points,
                kclusters=num_clusters[0],
                niter=10,
                dist=dist,
                kmeanspp_init=True,
            )

        for level in range(1, len(num_clusters)):
            landmarks, cluster_cat = kmeans_hier_inner(
                embeddings_cat,
                cluster_cat,
                num_clusters[level],
                num_clusters_above[level - 1],
                niter=10,
                dist=dist,
            )
        cluster_cat.unsqueeze_(1)
        if separate:
            num_clusters_flat = torch.tensor(num_clusters).prod()
            landmarks = landmarks.reshape(
                len(embeddings), num_hashes, num_clusters_flat, emb_size
            )
    elif method == "kmeans":
        landmarkss = []
        cluster_cats = []
        for _ in range(num_hashes):
            if separate:
                landmarks, cluster_cat = kmeans(
                    embeddings_cat,
                    num_points,
                    kclusters=num_clusters,
                    niter=10,
                    dist=dist,
                    kmeanspp_init=True,
                )
            else:
                landmarks, cluster_cat = kmeans_cat(
                    embeddings_cat,
                    num_points,
                    kclusters=num_clusters,
                    niter=10,
                    dist=dist,
                    kmeanspp_init=True,
                )
            landmarkss.append(landmarks)
            cluster_cats.append(cluster_cat)
        landmarks = torch.stack(landmarkss, dim=1)
        cluster_cat = torch.stack(cluster_cats, dim=1)
        del landmarkss, cluster_cats
    elif method == "angular_lsh":
        landmarks = None
        cluster_cat = get_angular_buckets(
            embeddings_cat, num_points, num_buckets=num_clusters, num_hashes=num_hashes
        )
    else:
        raise ValueError(f"Unknown landmarks/clustering method: '{method}'")

    if return_assignments:
        if cluster_cat is None:
            if separate:
                assert num_hashbands == 1
                assert num_hashes_per_band == 1
                landmarks = landmarks.reshape(
                    len(embeddings), num_hashes, num_clusters, emb_size
                )
                cluster = [
                    get_cluster_assignments(
                        embeddings[i],
                        landmarks[i],
                        mask_value=num_clusters,
                        dist=dist,
                        num_points=num_points[i],
                    )
                    for i in range(2)
                ]
            else:
                cluster = [
                    get_cluster_assignments(
                        embeddings[i],
                        landmarks,
                        mask_value=num_clusters,
                        dist=dist,
                        num_points=num_points[i],
                    )
                    for i in range(2)
                ]
        else:
            if separate:
                assert num_hashbands == 1
                assert num_hashes_per_band == 1
                cluster = list(torch.split(cluster_cat, batch_size, dim=0))
            else:
                cluster = list(torch.split(cluster_cat, max_points, dim=-1))

        if num_hashes_per_band > 1:
            for i in range(len(cluster)):
                assert batch_size == 1  # Probably only works without padding
                cluster[i] = cluster[i].reshape(
                    batch_size, num_hashbands, num_hashes_per_band, max_points
                )
                factor = num_clusters ** torch.arange(
                    num_hashes_per_band,
                    dtype=cluster[i].dtype,
                    device=cluster[i].device,
                )[None, None, :, None]
                cluster[i] = torch.sum(factor * cluster[i], dim=-2)

        return landmarks, cluster
    else:
        return landmarks
