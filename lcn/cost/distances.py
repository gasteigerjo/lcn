from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Distance(ABC):
    @abstractmethod
    def norm(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def cdist(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pairwise_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass


class PNorm(Distance):
    def __init__(self, p=2):
        self.p = p

    def norm(self, x):
        return torch.norm(x, p=self.p, dim=-1)

    def cdist(self, x1, x2):
        batch_size = x1.shape[0]
        max_cdist_batch = 1_048_576  # = 2^20, 2^21 = 2_097_152
        if batch_size * x1.shape[1] * x2.shape[1] >= max_cdist_batch:
            dists = []
            if x1.shape[1] * x2.shape[1] >= max_cdist_batch:
                chunk = max_cdist_batch // x2.shape[1]
                for b in range(batch_size):
                    dists_inner = []
                    for start1 in range(0, x1.shape[1], chunk):
                        dists_inner.append(
                            self.cdist(
                                x1[b : b + 1, start1 : start1 + chunk], x2[b : b + 1]
                            )
                        )
                    dists.append(torch.cat(dists_inner, dim=1))
            else:
                chunk = max_cdist_batch // (x1.shape[1] * x2.shape[1])
                for start0 in range(0, batch_size, chunk):
                    dists.append(
                        self.cdist(
                            x1[start0 : start0 + chunk], x2[start0 : start0 + chunk]
                        )
                    )
            return torch.cat(dists, dim=0)
        else:
            return torch.cdist(x1, x2, p=self.p)

    def pairwise_distance(self, x1, x2):
        return self.norm(x1 - x2)


class ExpPNorm(Distance):
    def __init__(self, p, lmbda=1):
        self.pnorm = PNorm(p)
        self.lmbda = lmbda

    def norm(self, x):
        return torch.exp(self.lmbda * self.pnorm.norm(x))

    def cdist(self, x1, x2):
        return torch.exp(self.lmbda * self.pnorm.cdist(x1, x2))

    def pairwise_distance(self, x1, x2):
        return self.norm(x1 - x2)


class Cosine(Distance):
    def norm(self, x):
        return torch.ones(x.shape, dtype=x.dtype, device=x.device)

    def cdist(self, x1, x2, eps=1e-8):
        x1_norm = torch.norm(x1, p=2, dim=-1)
        x2_norm = torch.norm(x2, p=2, dim=-1)
        dot = x1 @ x2.transpose(-1, -2)
        cos_sim = dot / torch.clamp(x1_norm[:, :, None] * x2_norm[:, None, :], min=eps)
        return torch.sqrt(torch.clamp(1 - cos_sim, min=0))

    def pairwise_distance(self, x1, x2, eps=1e-8):
        return torch.sqrt(1 - F.cosine_similarity(x1, x2, dim=-1, eps=eps))


class AbsCosine(Distance):
    def norm(self, x):
        return torch.ones(x.shape, dtype=x.dtype, device=x.device)

    def cdist(self, x1, x2, eps=1e-8):
        x1_norm = torch.norm(x1, p=2, dim=-1)
        x2_norm = torch.norm(x2, p=2, dim=-1)
        dot = x1 @ x2.transpose(-1, -2)
        cos_sim = dot / torch.clamp(x1_norm[:, :, None] * x2_norm[:, None, :], min=eps)
        return torch.sqrt(torch.clamp(1 - torch.abs(cos_sim), min=0))

    def pairwise_distance(self, x1, x2, eps=1e-8):
        return torch.sqrt(1 - torch.abs(F.cosine_similarity(x1, x2, dim=-1, eps=eps)))


class Dot(Distance):
    def norm(self, x):
        return torch.norm(x, p=2, dim=-1)

    def cdist(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        dot = x1 @ x2.transpose(-1, -2)
        return torch.sqrt(x1_norm[:, :, None] + x2_norm[:, None, :] - dot)

    def pairwise_distance(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        dot = (x1 * x2).sum(-1)
        return torch.sqrt(x1_norm + x2_norm - dot)


class AbsDot(Distance):
    def norm(self, x):
        return torch.norm(x, p=2, dim=-1)

    def cdist(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        dot = x1 @ x2.transpose(-1, -2)
        return torch.sqrt(x1_norm[:, :, None] + x2_norm[:, None, :] - torch.abs(dot))

    def pairwise_distance(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        dot = (x1 * x2).sum(-1)
        return torch.sqrt(x1_norm + x2_norm - torch.abs(dot))


class KernelDist(Distance):
    def __init__(self, kernel, transform_fn=lambda x: -x):
        self.kernel = kernel
        self.transform_fn = transform_fn

    def norm(self, x):
        return self.transform_fn(self.kernel.csim(x, x))

    def cdist(self, x1, x2):
        return self.transform_fn(self.kernel.csim(x1, x2))

    def pairwise_distance(self, x1, x2):
        return self.transform_fn(self.kernel.pairwise_similarity(x1, x2))
