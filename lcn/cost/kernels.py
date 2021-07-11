from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Kernel(ABC):
    @abstractmethod
    def csim(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pairwise_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass


class Laplacian(Kernel):
    def __init__(self, dist, lmbda=1):
        self.dist = dist
        self.lmbda = lmbda

    def csim(self, x1, x2):
        if type(self.lmbda) is torch.Tensor and self.lmbda.dim() == 1:
            lmbda = self.lmbda[:, None, None]
        else:
            lmbda = self.lmbda
        return torch.exp(-lmbda * self.dist.cdist(x1, x2))

    def pairwise_similarity(self, x1, x2):
        if type(self.lmbda) is torch.Tensor and self.lmbda.dim() == 1:
            lmbda = self.lmbda[:, None]
        else:
            lmbda = self.lmbda
        return torch.exp(-lmbda * self.dist.pairwise_distance(x1, x2))


class Attention(Kernel):
    """
    Attention with softmax. Not stable like this.
    """

    def csim(self, x1, x2):
        return torch.exp(x1 @ x2.transpose(-1, -2))

    def pairwise_similarity(self, x1, x2):
        return torch.exp((x1 * x2).sum(-1))


class Dot(Kernel):
    def __init__(self, lmbda=1):
        self.lmbda = lmbda

    def csim(self, x1, x2):
        return self.lmbda * (x1 @ x2.transpose(-1, -2))

    def pairwise_similarity(self, x1, x2):
        return self.lmbda * (x1 * x2).sum(-1)


class Cosine(Kernel):
    def csim(self, x1, x2, eps=1e-8):
        x1_norm = torch.norm(x1, p=2, dim=-1)
        x2_norm = torch.norm(x2, p=2, dim=-1)
        dot = x1 @ x2.transpose(-1, -2)
        return dot / torch.clamp(x1_norm[:, :, None] * x2_norm[:, None, :], min=eps)

    def pairwise_similarity(self, x1, x2, eps=1e-8):
        return F.cosine_similarity(x1, x2, dim=-1, eps=eps)


class DistanceKernel(Kernel):
    def __init__(self, dist):
        super().__init__()

        self.dist = dist

    def csim(self, x1, x2):
        return -self.dist.cdist(x1, x2)

    def pairwise_similarity(self, x1, x2):
        return -self.dist.pairwise_distance(x1, x2)
